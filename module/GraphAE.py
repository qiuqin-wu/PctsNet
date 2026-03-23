import random
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
torch.cuda.empty_cache()
# 全局种子设置 (通常在代码开头)
#SEED = 500
#random.seed(SEED)         # Python内置随机模块
#np.random.seed(SEED)      # NumPy随机生成器
#torch.manual_seed(SEED)   # PyTorch随机种子


from scipy.spatial import distance
import scipy.sparse as sp
import networkx as nx

import gae.utils as gae_util
from gae.optimizer import loss_function
from gae.layers import GraphConvolution

import time
import os
import pickle

import plots

from module.Arc import Architecture
from scipy.sparse import csr_matrix
from utils import set_random_seed




class GraphAE(Architecture):
    def __init__(self, inData, epoch,seed, exp_dirs,contrastive_weight=0.1,margin=1.0,lambda_feat=1.0,beta=1.0,lr=1e-2,hidden1_dim=128,hidden2_dim=64):
        self.seed = seed
        set_random_seed(seed)
        self.inData = inData
        self.epoch = epoch

        #保存参数
        self.contrastive_weight = contrastive_weight
        self.margin = margin
        self.lambda_feat = lambda_feat
        self.beta = beta
        self.lr = lr
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        #初始化模型
        #self.model = GAE(inData.shape[1]).to(self.param['device'])
        self.model = GAE(inData.shape[1],  hidden1_dim=self.hidden1_dim, hidden2_dim=self.hidden2_dim,feat_dim=inData.shape[1], lambda_feat=self.lambda_feat).to(self.param['device'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.exp_dirs = exp_dirs

        self.loss_ls = []
        self.tolerance = 0


    #正负样本对
    def generate_pos_neg_pairs(self, adj, num_neg=1):
        N = adj.shape[0]
        pos_pairs, neg_pairs = [], []
        pos_edges = torch.nonzero(adj > 0, as_tuple=False).tolist()
        for i, j in pos_edges:
            if i < j:
                pos_pairs.append((i, j))

        while len(neg_pairs) < len(pos_pairs) * num_neg:
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            if i != j and adj[i, j] == 0:
                neg_pairs.append((i, j))
        return pos_pairs, neg_pairs

    #对比损失

    def contrastive_loss(self, z, pos_pairs, neg_pairs):
        loss = 0.0
        for i, j in pos_pairs:
            D = torch.norm(z[i] - z[j], p=2)
            loss += D.pow(2)

        for i, j in neg_pairs:
            D = torch.norm(z[i] - z[j], p=2)
            loss += F.relu(self.margin - D).pow(2)

        loss = loss / (len(pos_pairs) + len(neg_pairs) + 1e-6)
        return loss

    def run(self):
        print('### Starting Graph AE')

        # 返回参数太多了，待分解
        X_embed_normalized, CCC_graph_edge_index, CCC_graph, norm, pos_weight, edgeList, adj, adj_no_normalization,test_edges, test_edges_false = self.getGraph(self.inData)
        #embed, recon_graph = self.train(X_embed_normalized, CCC_graph_edge_index, CCC_graph, norm, pos_weight)
        embed, gae_info, recon_graph, x_hat = self.train(X_embed_normalized, CCC_graph_edge_index, CCC_graph, norm, pos_weight)

        # edgeList added just for benchmark testing
        # === 添加 ROC / AP 测试 ===
        adj_orig = adj.copy()
        adj_orig.setdiag(0)
        adj_orig.eliminate_zeros()
        
        roc_save_path = os.path.join(self.exp_dirs, f"epoch{self.epoch}_roc_curve.png")
        roc_score, ap_score = gae_util.get_roc_score(embed.detach().cpu().numpy(), adj_orig, test_edges, test_edges_false, plot=True,save_path=self.exp_dirs)
        print(f"[GraphAE ROC/AP] AUC: {roc_score:.4f}, AP: {ap_score:.4f}")
        
        result = (embed.detach().cpu().numpy(), edgeList, adj_no_normalization)
        self.save(result)
        return result

    def save(self, data):
        filePath = os.path.join(self.exp_dirs, f"epoch{self.epoch}-graphAE")
        with open(filePath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, epoch):
        filePath = os.path.join(self.exp_dirs, f"epoch{self.epoch}-graphAE")
        with open(filePath, 'rb') as f:
            data = pickle.load(f)
        return data

    def train(self, X_embed_normalized, CCC_graph_edge_index, CCC_graph, norm, pos_weight):
        self.model.train()
        for epoch in range(self.param['graph_AE_epoch']):
            t = time.time()

            self.optimizer.zero_grad()
            embed, gae_info, recon_graph,x_hat = self.model(X_embed_normalized,
                                                      CCC_graph_edge_index)
            # 生成正负样本对
            pos_pairs, neg_pairs = self.generate_pos_neg_pairs(CCC_graph)
            # 计算总损失（GAE + 特征 + KL + 对比）
            loss,L_GAE,L_feat,L_KL = loss_function(preds=recon_graph,
                                 labels=CCC_graph,
                                 mu=gae_info[0], logvar=gae_info[1],
                                 n_nodes=self.inData.shape[0],
                                 norm=norm,
                                 pos_weight=pos_weight,x_hat=x_hat,x=X_embed_normalized,lambda_feat=self.lambda_feat,beta=self.beta,embed=embed,pos_pairs=pos_pairs,neg_pairs=neg_pairs,contrastive_weight=self.contrastive_weight,margin=self.margin)  # KL 权重（可改）)
            #对比损失
            #pos_pairs, neg_pairs = self.generate_pos_neg_pairs(CCC_graph)
            #L_contrast = self.contrastive_loss(embed, pos_pairs, neg_pairs)
            #total_loss = loss + self.contrastive_weight * L_contrast

            #total_loss.backward()
            #cur_loss = total_loss.item()

            loss.backward()
            cur_loss = loss.item()
            self.loss_ls.append(cur_loss)
            self.optimizer.step()
            # print(f"----------------> Epoch: {epoch + 1}, train_loss_gae={cur_loss:.5f}, time={time.time() - t:.5f}")

            if self.checkStop():
                break
        plots.plot_loss(self.loss_ls, 'GAE', self.epoch, Architecture.exp_dirs)

        return embed, gae_info,recon_graph,x_hat

    def checkStop(self):
        if len(self.loss_ls) > 1 and self.loss_ls[-1] >= self.loss_ls[-2]:
            self.tolerance += 1
        if self.tolerance > 50:
            print('early stop!')
            return True
        return False

    def getGraph(self, X_embed):
        adj, adj_no_normalization,adj_train, edgeList, test_edges, test_edges_false = feature2adj(X_embed)
        adj_norm = gae_util.preprocess_graph(adj)
        adj_label = (adj_train + sp.eye(adj_train.shape[0])).toarray()

        #zDiscret = X_embed > np.mean(X_embed, axis=0)
        #zDiscret = 1.0 * zDiscret
        #X_embed_normalized = torch.from_numpy(zDiscret).type(torch.FloatTensor).to(self.param['device'])
        X_embed_normalized = torch.from_numpy((X_embed - X_embed.mean(axis=0)) / (X_embed.std(axis=0) + 1e-6)).type(torch.FloatTensor).to(self.param['device'])
        CCC_graph_edge_index = adj_norm.to(self.param['device'])
        # 这里直接赋值CCC_graph
        CCC_graph = torch.from_numpy(adj_label).type(torch.FloatTensor).to(self.param['device'])

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        return X_embed_normalized, CCC_graph_edge_index, CCC_graph, norm, pos_weight, edgeList, adj,adj_no_normalization,test_edges, test_edges_false



class FeatureDecoder(nn.Module):
    """Decoder for reconstructing input features X"""
    def __init__(self, hidden_dim, input_dim):
        super(FeatureDecoder, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)   # 输出维度和原输入一致
        )

    def forward(self, z):
        x_hat = self.mlp(z) #使用原变量名 x_hat
        return x_hat



class GAE(nn.Module):
    def __init__(self, dim_in,hidden1_dim=128, hidden2_dim=64, feat_dim=None, lambda_feat=1.0):
        super(GAE, self).__init__()
        print("dim_in:",dim_in)
        self.gc1 = GraphConvolution(dim_in, hidden1_dim, 0, act=F.relu)
        self.gc2 = GraphConvolution(hidden1_dim, hidden2_dim, 0, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden1_dim, hidden2_dim, 0, act=lambda x: x)

        self.decode = InnerProductDecoder(0, act=lambda x: x)

        #特征解码器，保持原变量名 x_hat
        if feat_dim is not None:
            self.feat_decode = FeatureDecoder(hidden2_dim, feat_dim)
        else:
            self.feat_decode = None

        #λ 超参数保持不变
        self.lambda_feat = lambda_feat

    def encode_gae(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, in_nodes_features, edge_index):
        # [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        

        gae_info = self.encode_gae(in_nodes_features, edge_index)
        out_nodes_features = self.reparameterize(*gae_info)

        recon_graph = self.decode(out_nodes_features)
        #如果有 FeatureDecoder，额外输出 x_hat
        x_hat = self.feat_decode(out_nodes_features) if self.feat_decode is not None else None
        return out_nodes_features, gae_info, recon_graph,x_hat


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


def feature2adj(X_embed):
    edgeList = calculateKNNgraphDistanceMatrixStatsSingleThread(X_embed)
    graphdict = edgeList2edgeDict(edgeList, X_embed.shape[0])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))  # 为邻接矩阵A
    adj_no_normalization=adj  # 为邻接矩阵A
    adj_orig = adj
    # 这两行好像是图网络标准操作
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape) #去除自环
    adj_orig.eliminate_zeros()  # 标准化后的邻接矩阵L0
    #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = gae_util.mask_test_edges(adj)
    adj_train, train_edges,  test_edges, test_edges_false = gae_util.mask_test_edges(adj)
    
    adj = adj_train

    return adj, adj_no_normalization,adj_train, edgeList, test_edges, test_edges_false


def calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10):
    r"""
    Thresholdgraph: KNN Graph with stats one-std based methods, SingleThread version
    """

    edgeList = []

    p_time = time.time()
    for i in np.arange(featureMatrix.shape[0]):
        tmp = featureMatrix[i, :].reshape(1, -1)
        # 得到第i个点到所有点的距离值
        distMat = distance.cdist(tmp, featureMatrix, distanceType)
        # 前k+1个最小的距离的index
        res = distMat.argsort()[:k + 1]
        tmpdist = distMat[0, res[0][1:k + 1]]  # 存储前k个最小的距离值，去除了自身的情况
        # 不太明白为什么boundary为什么要这样确定
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k + 1):
            # TODO: check, only exclude large outliners
            #if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
            if distMat[0, res[0][j]] <= boundary:
                #weight = 1.0
                weight = np.exp(-distMat[0, res[0][j]])  # 使用exp来计算权重
            else:
                weight = 0.0
            edgeList.append((i, res[0][j], weight))

    print("KNN run time: ", time.time() - p_time)
    # 我发现第一次得到的edgeList的元素个数是29470，也是就是说每个点和最近的10个点都有连边
    # 需要检测一下随着迭代edgeList怎么变化
    return edgeList


def edgeList2edgeDict(edgeList, nodesize):
    # graphdict保存的是key是所有顶点，value是顶点所有一阶邻居的list
    graphdict = {}
    tdict = {}

    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

    # check and get full matrix
    # 考虑到有些顶点没有邻居的情况
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []

    return graphdict


def normalize_features_dense(node_features_dense):
    assert isinstance(node_features_dense, np.ndarray), f'Expected np matrix got {type(node_features_dense)}.'

    # The goal is to make feature vectors normalized (sum equals 1), but since some feature vectors are all 0s
    # in those cases we'd have division by 0 so I set the min value (via np.clip) to 1.
    # Note: 1 is a neutral element for division i.e. it won't modify the feature vector
    return node_features_dense / np.clip(node_features_dense.sum(1, keepdims=True), a_min=1, a_max=None)


def convert_adj_to_edge_index(adjacency_matrix):
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
                edge_index.append([src_node_id, trg_nod_id])

    return np.asarray(edge_index).transpose()  # change shape from (N,2) -> (2,N)

