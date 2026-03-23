import pickle
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score

import os

import networkx as nx
from igraph import * # ignore the squiggly underline, not an error

from module.Arc import Architecture
from utils import set_random_seed


class Cluster:
    def __init__(self,inData,epoch,metadata,exp_dirs,seed):
        self.seed = seed
        set_random_seed(seed)
        print(f"Initializing Cluster with epoch={epoch} and metadata keys: {metadata.keys()}")
        # edgeList = (cell_i, cell_a), (cell_i, cell_b), ...
        self.graph_emb,self.edgeList=inData
        self.epoch=epoch
        self.true_labels=metadata['label']
        self.exp_dirs = exp_dirs  # 保存实验结果目录路径
        #self.true_labels=self.dataUpdated['true_labels']

        #pass

    def run(self):
        print('--------> Start Clustering ...')
        listResult, size = self.generateLouvainCluster()  

        k = len(np.unique(listResult))
        print(f'----------------> Louvain clusters count: {k}')
        
        resolution =  0.8 if self.graph_emb.shape[0] < 2000 else 0.4 # based on num of cells
        k = int(k * resolution) if int(k * resolution) >= 3 else 2

        clustering = KMeans(n_clusters=k, random_state=self.seed).fit(self.graph_emb)  # 输入k，再利用KMeans算法聚类一次，得到聚类类别及内容
        listResult = clustering.predict(self.graph_emb) # (n_samples,) Index of the cluster each sample belongs to.

        if len(set(listResult)) > 30 or len(set(listResult)) <= 1:
            print(f"----------------> Stopping: Number of clusters is {len(set(listResult))}")
            listResult = trimClustering(
                listResult, minMemberinCluster=5, maxClusterNumber=30)

        print(f'----------------> Total Cluster Number: {len(set(listResult))}')
        # tuple{'ct_list', 'lists_of_idx'}
        silhouette_avg = silhouette_score(self.graph_emb, listResult)
        print(f"Silhouette Score_overall: {silhouette_avg:.4f}")
        result=self.cluster_output_handler(listResult)
        self.save(result)
        self.eval(result[0])
        ARI=self.eval(result[0])
        return result,ARI


    def run_choose_best(self):
        print('### Start Clustering ...')
        t0=time.time()

        max_k=0
        max_silhouette_avg=0
        max_listResult=None
        for k in range(5,20):
            clustering = KMeans(n_clusters=k, random_state=self.seed).fit(self.graph_emb)  # 输入k，再利用KMeans算法聚类一次，得到聚类类别及内容
            listResult = clustering.predict(self.graph_emb) # (n_samples,) Index of the cluster each sample belongs to.
            #listResult = np.ravel(listResult)  # 确保 listResult 是一维数组
            silhouette_avg = silhouette_score(self.graph_emb, listResult)
            if silhouette_avg>max_silhouette_avg:
                max_k=k
                max_silhouette_avg=silhouette_avg
                max_listResult=listResult
            print(k,silhouette_avg)

        print(f"the best one: k={max_k}, silhouette score: {max_silhouette_avg}")
        print(f'Total Cluster Number: {len(set(max_listResult))}')
        # tuple{'ct_list', 'lists_of_idx'}
        result=self.cluster_output_handler(max_listResult)
        self.save(result)
        self.eval(result[0])
        ARI=self.eval(result[0])
        return result,ARI
        print("用时：",time.time()-t0)
       

    def eval(self,pre_labels):
        print("pre_labels.shape:",pre_labels.shape)
        print("true_labels.shape:",self.true_labels.shape)
        ARI=adjusted_rand_score(self.true_labels,pre_labels)
        NMI = normalized_mutual_info_score(self.true_labels, pre_labels)
        AMI = adjusted_mutual_info_score(self.true_labels, pre_labels)
        homogeneity = homogeneity_score(self.true_labels, pre_labels)
        print(f'ARI:{ARI}')
        print(f'NMI: {NMI}')
        print(f"AMI: {AMI:.4f}")
        print(f"Homogeneity Score: {homogeneity:.4f}")
        return ARI
        

    def save(self,data):
        filePath=os.path.join(self.exp_dirs,f"epoch{self.epoch}-Cluster")
        with open(filePath,'wb+') as f:
            pickle.dump(data,f)


    def load(self,epoch):
        filePath=os.path.join(self.exp_dirs,f"epoch{self.epoch}-Cluster")
        with open(filePath,'rb') as f:
            data=pickle.load(f)
        return data

    def generateLouvainCluster(self):
        """
        Louvain Clustering using igraph
        """
        Gtmp = nx.Graph()
        Gtmp.add_weighted_edges_from(self.edgeList)
        W = nx.adjacency_matrix(Gtmp)
        W = W.todense()
        graph = Graph.Weighted_Adjacency(
            W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False) # ignore the squiggly underline, not errors
        louvain_partition = graph.community_multilevel(
            weights=graph.es['weight'], return_levels=False)
        size = len(louvain_partition)
        hdict = {}
        count = 0
        for i in range(size):
            tlist = louvain_partition[i]
            for j in range(len(tlist)):
                hdict[tlist[j]] = i
                count += 1

        listResult = []
        for i in range(count):
            listResult.append(hdict[i])

        return listResult, size

    def cluster_output_handler(self,listResult):
        clusterIndexList = []
        for i in range(len(set(listResult))):
            clusterIndexList.append([])
        for i in range(len(listResult)):
            clusterIndexList[listResult[i]].append(i)

        print("cell counts:",[len(i) for i in clusterIndexList])
        print("Before dischard:",len(clusterIndexList))
        # 丢弃细胞数少于50的簇
        #clusterIndexList=self.discard(clusterIndexList)
        #print("After dischard:",len(clusterIndexList))
        return np.array(listResult), clusterIndexList
    
    def discard(self,lists_of_idx,threshold=50):
        result=[i for i in lists_of_idx if len(i)>=threshold]
        return result
    
    def train(self):
        pass



def convert_adj_to_edge_list(adjacency_matrix):
    """
    """
    assert isinstance(adjacency_matrix, np.ndarray), f'Expected NumPy array got {type(adjacency_matrix)}.'
    height, width = adjacency_matrix.shape
    assert height == width, f'Expected square shape got = {adjacency_matrix.shape}.'

    # If there are infs that means we have a connectivity mask and 0s are where the edges in connectivity mask are,
    # otherwise we have an adjacency matrix and 1s symbolize the presence of edges.
    # active_value = 0 if np.isinf(adjacency_matrix).any() else 1

    edge_index = []
    for src_node_id in range(height):
        for trg_nod_id in range(width):
            if adjacency_matrix[src_node_id, trg_nod_id] > 0:
                edge_index.append((src_node_id, trg_nod_id, adjacency_matrix[src_node_id, trg_nod_id]))

    return np.asarray(edge_index)  # (N,3)


    
    # {
    #     'ct_list': listResult,
    #     'lists_of_idx' : clusterIndexList
    # }

def trimClustering(listResult, minMemberinCluster=5, maxClusterNumber=30):
    '''
    If the clustering numbers larger than certain number, use this function to trim. May have better solution
    '''
    numDict = {}
    for item in listResult:
        if not item in numDict:
            numDict[item] = 0
        else:
            numDict[item] = numDict[item]+1

    size = len(set(listResult))
    changeDict = {}
    for item in range(size):
        if numDict[item] < minMemberinCluster or item >= maxClusterNumber:
            changeDict[item] = ''

    count = 0
    for item in listResult:
        if item in changeDict:
            listResult[count] = maxClusterNumber
        count += 1

    return listResult


if __name__ == '__main__':
    # 调试该文件
    import pickle
    with open("./debug/graph_AE",'rb') as f:
        graph_embed, edgeList=pickle.load(f)

    # cluster_labels, cluster_lists_of_idx = clustering_handler(graph_embed, edgeList) 