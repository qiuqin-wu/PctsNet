import pickle
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score
from sklearn.metrics.pairwise import cosine_similarity

import os

import networkx as nx
from igraph import * # ignore the squiggly underline, not an error

from module.Arc import Architecture
from utils import set_random_seed

import community as community_louvain

import igraph as ig
import leidenalg

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

    def run(self, resolution=1.0):
        print('--------> Start Clustering ...')
        listResult, size = self.generateLouvainCluster(resolution=resolution)  
        #listResult, size = self.generateLeidenCluster(resolution=1.0)
        k = len(np.unique(listResult))
        print(f'----------------> Louvain clusters count: {k}')
        
        #resolution =  0.8 if self.graph_emb.shape[0] < 2000 else 0.4 # based on num of cells
        #k = int(k * resolution) if int(k * resolution) >= 3 else 2

        #clustering = KMeans(n_clusters=k, random_state=self.seed).fit(self.graph_emb)  # 输入k，再利用KMeans算法聚类一次，得到聚类类别及内容
        #listResult = clustering.predict(self.graph_emb) # (n_samples,) Index of the cluster each sample belongs to.
        '''
        if len(set(listResult)) > 30 or len(set(listResult)) <= 1:
            print(f"----------------> Stopping: Number of clusters is {len(set(listResult))}")
            listResult = trimClustering(
                listResult, minMemberinCluster=5, maxClusterNumber=30)
        '''
        print(f'----------------> Total Cluster Number: {len(set(listResult))}')
        # tuple{'ct_list', 'lists_of_idx'}
        silhouette_avg = silhouette_score(self.graph_emb, listResult)
        print(f"Silhouette Score_overall: {silhouette_avg:.4f}")
        result=self.cluster_output_handler(listResult)
        self.save(result)
        #self.eval(result[0])
        #ARI=self.eval(result[0])
        #return result,ARI
        return result

    '''
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
    '''

    def run_choose_best(self):
        print('### Start Clustering ...')
        t0=time.time()

        listResult, size = self.generateLouvainCluster()
        silhouette_avg = silhouette_score(self.graph_emb, listResult)
        print(f"Louvain clusters: {len(set(listResult))}, silhouette score: {silhouette_avg:.4f}")
        result = self.cluster_output_handler(listResult)
        self.save(result)
        ARI = self.eval(result[0])
        print("用时：", time.time()-t0)
        return result, ARI

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

    def generateLouvainCluster(self,resolution=1.0):
        """
        Louvain Clustering using igraph
        """
        #Gtmp = nx.Graph()
        #Gtmp.add_weighted_edges_from(self.edgeList)
        #W = nx.adjacency_matrix(Gtmp)
        #W = W.todense()
        #graph = Graph.Weighted_Adjacency(
           # W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False) # ignore the squiggly underline, not errors
        #louvain_partition = graph.community_multilevel(
           # weights=graph.es['weight'], resolution_parameter=resolution,return_levels=False)
        #size = len(louvain_partition)
        #hdict = {}
        #count = 0
        '''
        for i in range(size):
            tlist = louvain_partition[i]
            for j in range(len(tlist)):
                hdict[tlist[j]] = i
                count += 1
        listResult = []
        for i in range(count):
            listResult.append(hdict[i])
        '''
        '''
        G = nx.Graph()
        G.add_weighted_edges_from(self.edgeList)

        partition = community_louvain.best_partition(G, resolution=resolution, weight='weight')
        n_nodes = G.number_of_nodes()
        listResult = [partition[i] for i in range(n_nodes)]

        size = len(set(listResult))

        return listResult, size
        '''
        similarity = cosine_similarity(self.graph_emb)
        similarity[similarity < 0] = 0.0
        G = nx.from_numpy_array(similarity)
        partition = community_louvain.best_partition(G, resolution=resolution,weight='weight')
        n_nodes = G.number_of_nodes()
        listResult = [partition[i] for i in range(n_nodes)]
        size = len(set(listResult))
        return listResult, size

    def generateLeidenCluster(self, resolution=1.0):
        '''
        用 Leiden 算法进行社区检测
        '''
        # 1) 构建 igraph Graph
        # edgeList 形式: (src, dst, weight)
        edges = [(int(i), int(j)) for i, j, w in self.edgeList]
        weights = [float(w) for i, j, w in self.edgeList]
        n_nodes = len(self.graph_emb)  # 总节点数
        G = ig.Graph(edges=edges, directed=False)
        G.vs["name"] = list(range(n_nodes))
        # 2) Leiden 聚类
        partition = leidenalg.find_partition(
                G,
                leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                resolution_parameter=resolution
        )

        # 3) 输出 cluster label
        listResult = np.zeros(n_nodes, dtype=int)
        for cluster_id, cluster_nodes in enumerate(partition):
            for node in cluster_nodes:
                listResult[node] = cluster_id

        size = len(set(listResult))
        return listResult, size












    def cluster_output_handler(self,listResult):
        # 重新映射成连续整数
        unique_ids = sorted(set(listResult))
        id_map = {old: new for new, old in enumerate(unique_ids)}
        listResult = np.array([id_map[c] for c in listResult])
        clusterIndexList = [[] for _ in range(len(unique_ids))]
        for i in range(len(listResult)):
            clusterIndexList[listResult[i]].append(i)

        print("cell counts:", [len(i) for i in clusterIndexList])
        print("Before discard:", len(clusterIndexList))
        #丢弃小簇

        #clusterIndexList = self.discard(clusterIndexList)
        clusterIndexList = self.merge_small_clusters(clusterIndexList, threshold=0)
        print("After discard:", len(clusterIndexList))


        #保留大簇索引
        valid_indices = [idx for clust in clusterIndexList for idx in clust]
        filtered_listResult = listResult[valid_indices]

        #重新映射clusetr id 为连续整数

        unique_filtered = sorted(set(filtered_listResult))
        id_map_filtered = {old: new for new, old in enumerate(unique_filtered)}
        filtered_listResult = np.array([id_map_filtered[c] for c in filtered_listResult])

        #计算ARI

        if self.true_labels is not None:
            y_true = np.array(self.true_labels)[valid_indices]
            ari = adjusted_rand_score(y_true, filtered_listResult)
            print(f"ARI after removing small clusters: {ari:.4f}")
            NMI = normalized_mutual_info_score(y_true, filtered_listResult)
            print(f"NMI after removing small clusters: {NMI:.4f}")
            AMI = adjusted_mutual_info_score(y_true, filtered_listResult)
            print(f"AMI after removing small clusters: {AMI:.4f}")
            homogeneity = homogeneity_score(y_true, filtered_listResult)
            print(f"Homogeneity after removing small clusters: {homogeneity:.4f}")

            # silhouette 需要用 embedding + 预测标签
            if len(set(filtered_listResult)) > 1:  # 必须至少有2个簇
                silhouette_avg = silhouette_score(self.graph_emb[valid_indices], filtered_listResult)
                print(f"Silhouette Score after removing small clusters: {silhouette_avg:.4f}")
            else:
                silhouette_avg = None
                print("Silhouette Score cannot be computed (only one cluster).")

        return filtered_listResult, clusterIndexList,ari







        #return listResult, clusterIndexList

        #clusterIndexList = []
        #for i in range(len(set(listResult))):
            #clusterIndexList.append([])
        #for i in range(len(listResult)):
            #clusterIndexList[listResult[i]].append(i)

        #print("cell counts:",[len(i) for i in clusterIndexList])
        #print("Before dischard:",len(clusterIndexList))
        # 丢弃细胞数少于50的簇
        #clusterIndexList=self.discard(clusterIndexList)
        #print("After dischard:",len(clusterIndexList))
        #return np.array(listResult), clusterIndexList
    
    def discard(self,lists_of_idx,threshold=50):
        
        #"删除小簇"
        #result=[i for i in lists_of_idx if len(i)>=threshold]
        #return result
        return [clust for clust in lists_of_idx if len(clust) >= threshold]
    

    def merge_small_clusters(self, lists_of_idx, threshold=50):
        '''
        把小簇并入最近的大簇，而不是直接丢掉
        '''
        clusterIndexList = lists_of_idx
        # 1) 计算每个簇的质心
        centroids = []
        for clust in clusterIndexList:
            emb = self.graph_emb[clust]  # 取出该簇所有细胞的embedding
            centroid = np.mean(emb, axis=0)
            centroids.append(centroid)

        # 2) 大簇和小簇分类
        big_clusters = [i for i, clust in enumerate(clusterIndexList) if len(clust) >= threshold]
        small_clusters = [i for i, clust in enumerate(clusterIndexList) if len(clust) < threshold]
        # 3) 把小簇分配到最近的大簇
        for s in small_clusters:
            if not big_clusters:  # 如果没有大簇，就保留原簇
                continue
            # 计算与所有大簇质心的距离
            dists = [np.linalg.norm(centroids[s] - centroids[b]) for b in big_clusters]
            nearest_big = big_clusters[np.argmin(dists)]
            # 合并
            clusterIndexList[nearest_big].extend(clusterIndexList[s])
            clusterIndexList[s] = []  # 清空小簇

        # 4) 去掉空簇
        clusterIndexList = [clust for clust in clusterIndexList if len(clust) > 0]

        return clusterIndexList














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
