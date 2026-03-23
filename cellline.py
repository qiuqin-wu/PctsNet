print('Loading Packages')

from module.Arc import Architecture
from module.GraphAE import GraphAE
from module.Cluster2 import Cluster
#from module.Cicero2 import Cicero
#from module.signac import Signac
from utils import set_random_seed

print("ss1")

import numpy as np
import pandas as pd
import argparse
import os
import utils
import pickle
import time
from scipy.sparse import csr_matrix
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

DEBUG = False
if not DEBUG:
    parser = argparse.ArgumentParser(description='This is description...')

    parser.add_argument('--name', type=str, default="cellline", help='help.')
    parser.add_argument('--cutoff', type=int, default=0, help='help.')  # select top 2000 col(i.e,peaks)
    parser.add_argument('--ccancutoff', type=float, default=0, help='help.')
    parser.add_argument('--seed', type=int, default=4500, help='Random seed')
    args, unknown = parser.parse_known_args()
    args = parser.parse_args()
    args.name = f"{args.name}_seed{args.seed}"
else:
    print("进入DEBUG模式！")


    class Args():
        def __init__(self):
            pass


    args = Args()
    args.name = "exp0"
    args.cutoff = 0
    args.ccancutoff = 0
# 打印设备信息（是否使用 GPU）
Architecture.print_device_info()
print("参数设置：")
print(args.name, args.cutoff, args.ccancutoff)

exp_dirs = f"/nvme2/wuqiuqin/compare/cellline2/results2/{args.name}"
if not os.path.exists(exp_dirs):
    os.makedirs(exp_dirs)
    print("创建了实验结果文件夹：", exp_dirs)
else:
    import shutil

    shutil.rmtree(exp_dirs)
    os.makedirs(exp_dirs)
    print("文件名重复，重新创建文件夹：", exp_dirs)

print('Preparing sc data')
#roc_save_path = os.path.join(exp_dirs, "roc_curve.png")
# 这个数据集有2947 cells, 489671 peaks
data_path = '/nvme2/wuqiuqin/compare/cellline/count_matrix_filter2.csv'
label_path = '/nvme2/wuqiuqin/compare/cellline/metadata_filter.tsv'

#总结果目录
base_dir = f"/nvme2/wuqiuqin/compare/cellline2/results2/{args.name}"
os.makedirs(base_dir, exist_ok=True)


Arc = Architecture(data_path, label_path, exp_dirs, args,seed=args.seed)

'''
df = pd.read_csv(data_path, index_col=0)  # 再次加载数据，不进行转置
print("df.shape:",df.shape)

print("args.cutoff:",args.cutoff)
if args.cutoff > 0:
   index=np.argsort(np.sum(df,1))[-args.cutoff:]
   df=df.iloc[index,:]



print("df.shape:",df.shape)
print(df.iloc[:5,:5])

row_names = df.index.tolist()  # 原始的行名
#col_names = df.columns.tolist()  # 原始的列名

col_names = [col.replace('.', '-') for col in df.columns]
df.columns=col_names

'''

col_names = Architecture.data['true_labels'][1].index.tolist()  # 549
row_names = Architecture.data['peak_names']  # 2000
col_names = [col.replace('.', '-') for col in col_names]

print(f"Row names: {row_names[:5]}")  # 查看前5个行名
print(f"Column names: {col_names[:5]}")  # 查看前5个列名

cell_peak_data = Architecture.data['expr']
print(type(cell_peak_data))

cell_peak_data = cell_peak_data.T
print(cell_peak_data.shape)
cell_peak_data_df = pd.DataFrame(cell_peak_data, index=row_names, columns=col_names)
cell_peak_data = cell_peak_data_df
cell_peak_data_df = cell_peak_data_df.T

metadata = Arc.getMetadata()  # 假设你在 Arc.py 中实现了这个方法来获取元数据
metadata.index = [idx.replace('.', '-') for idx in metadata.index]

print("haha1")



start_time = time.time()
#adj_old = None
#lam = 0.5
#converge_graphratio=0.01
print('Entering main loop')

print("cell_peak_data.shape: ", cell_peak_data.shape)

cell_peak_data = (cell_peak_data > 0).astype(int)
i = 0
#signac = Signac(cell_peak_data, metadata, epoch=i + 1, output_dir=exp_dirs)
#count_new, metadata_signac = signac.run()  # 获取更新后的数据

path2 = "/nvme2/wuqiuqin/compare/cellline/signac_reductions.csv"
path3 = "/nvme2/wuqiuqin/compare/cellline/signac_metadata.tsv"

count_new = pd.read_csv(path2, index_col=0)
metadata_signac = pd.read_csv(path3, sep='\t', index_col=0)


count_new_array = count_new.to_numpy()
count_new = count_new_array
print(f"Signac.run()后的耗时: {time.time() - start_time}秒")
print("metadata_signac.shape:", metadata_signac.shape)

print(type(count_new))
print("count_new.shape:", count_new.shape)
print("count_new[1:3]:", count_new[1:3, 1:3])
# 计算metadata行名在输入cell_peak_data（2034 cellls）中的索引
metadata_row_name = metadata_signac.index
cell_peak_data_col_name = cell_peak_data.columns
metadata_index = cell_peak_data_col_name.get_indexer(metadata_row_name)

#gae = GraphAE(count_new, i + 1,seed=args.seed,contrastive_weight=0.2,margin=1.0,lambda_feat=0.5,beta=2.0,lr=0.01,hidden1_dim=128,hidden2_dim=64)

param_grid = [{"contrastive_weight": 0.6, "margin": 2.0,"lambda_feat": 0.5, "beta":2.0, "lr":0.01, "hidden1_dim": 256, "hidden2_dim": 128},
        {"contrastive_weight": 0.6, "margin":1.0, "lambda_feat": 0.5, "beta":2.0, "lr":0.01,"hidden1_dim": 128, "hidden2_dim": 64},
        {"contrastive_weight": 0.6, "margin": 2.0, "lambda_feat": 0.5, "beta":2.0, "lr":0.01,"hidden1_dim": 128, "hidden2_dim": 64}]

#exp_name = (f"cw{params['contrastive_weight']}_m{params['margin']}_"
#            f"h1{params['hidden1_dim']}_h2{params['hidden2_dim']}"
#)
# 设置实验目录，包含参数
#exp_dirs2 = os.path.join(base_dir, exp_name)


for idx, params in enumerate(param_grid):
    #print(f"---- Running config {idx+1}: {params}")
    #每个参数组合单独目录
    exp_name = (
            f"cw{params['contrastive_weight']}_m{params['margin']}"
            f"_h1{params['hidden1_dim']}_h2{params['hidden2_dim']}"
    )
    exp_dirs2 = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dirs2, exist_ok=True)

    print(f"---- Running config {idx+1}: {params}, results -> {exp_dirs}")

    gae = GraphAE(
            count_new,
            i + 1,
            seed=args.seed,
            exp_dirs=exp_dirs2,
            contrastive_weight=params["contrastive_weight"],
            margin=params["margin"],
            lambda_feat=params["lambda_feat"],
            beta=params["beta"],
            lr=params["lr"],
            hidden1_dim=params["hidden1_dim"],
            hidden2_dim=params["hidden2_dim"],
    )
    emb2, edgeList,adj_no_normalization = gae.run()

    print(f"GraphAE.run()后的耗时: {time.time() - start_time}秒")
    print("GAE end")
    cluster = Cluster((emb2, edgeList), i + 1, metadata_signac, exp_dirs2,seed=args.seed)
    result = cluster.run(resolution=1.2)
    cluster_labels, cluster_lists_of_idx, ARI = result
    print("ARI_value:",ARI)
    print(f"Cluster.run_choose_best()后的耗时: {time.time() - start_time}秒")
    print("evalcluster end")


    



#emb2, edgeList,adj_no_normalization = gae.run()
    
''' 
    if i == 0:
        # 计算每个节点的度
        degree = np.array(adj_no_normalization.sum(axis=1)).flatten()  # 计算每个节点的度
        D_inv_sqrt = csr_matrix(np.diag(1.0 / np.sqrt(degree)))  # 计算 D^(-1/2)

        # 计算对称标准化邻接矩阵 A_laplacian
        A_laplacian = D_inv_sqrt @ adj_no_normalization @ D_inv_sqrt  # 矩阵乘法 L0
        print("First iteration: Calculated A_laplacian")

    adj_new = lam *A_laplacian + (1 -lam) * adj_no_normalization / np.sum(adj_no_normalization, axis=0)



    if adj_old is not None:
        graphChange = np.mean(abs(adj_new - adj_old))
        graphChangeThreshold = converge_graphratio * np.mean(abs(A_laplacian))

        if graphChange < graphChangeThreshold:
            print(f"Iteration {i}: Convergence reached, terminating the loop.")
            break  # 终止循环

    adj_old = adj_new    
'''
'''
print(f"GraphAE.run()后的耗时: {time.time() - start_time}秒")

print("GAE end")
cluster = Cluster((emb2, edgeList), i + 1, metadata_signac, exp_dirs,seed=args.seed)
#cluster_labels, cluster_lists_of_idx = cluster.run_choose_best()
result = cluster.run(resolution=1)
cluster_labels, cluster_lists_of_idx, ARI = result
print("ARI_value:",ARI)
#print("cluster end")
print(f"Cluster.run_choose_best()后的耗时: {time.time() - start_time}秒")
print("evalcluster end")
# break
'''
# 保存结果


# cluster_lists_of_idx0=[cluster_lists_of_idx[0]]
# cicero=Cicero(cluster_lists_of_idx0, i+1,cell_peak_data,metadata_signac,metadata_index)
    
'''   
    if i == 0:
        # 在 i == 0 时保存相关数据到 exp_dirs
        save_path = os.path.join(exp_dirs, f"epoch_{i}_initial_data.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump({
                "cluster_lists_of_idx": cluster_lists_of_idx,
                "cell_peak_data": cell_peak_data,
                "metadata_signac": metadata_signac,
                "metadata_index": metadata_index
            }, f)
        print(f"数据已保存到: {save_path}")






    cicero = Cicero(cluster_lists_of_idx, i + 1, cell_peak_data, metadata_signac, metadata_index,args.ccancutoff)
    cell_peak_data_new, peak_names, cell_indexs = cicero.run()
    print(f"Cicero.run()后的耗时: {time.time() - start_time}秒")
    print("本轮epoch结束\n")
    print("Cicero end")
    print("len(peak_names):", len(peak_names))
    print("cell_peak_data_new.shape:", cell_peak_data_new.shape)
    print("len(cell_indexs):", len(cell_indexs))
    print("nonzero_cell_peak_data_new:",(cell_peak_data_new !=0).sum().sum())

    # 将 cell_peak_data_df 的行名和列名转换为列表
    cell_peak_df_index = list(cell_peak_data_df.index)
    cell_peak_df_columns = list(cell_peak_data_df.columns)

    # 初始化一个与 cell_peak_data_df 大小相同的空矩阵，填充为 cell_peak_data_df 的值
    extended_matrix = cell_peak_data_df.values.copy()
    print("extended_matrix.shape:",extended_matrix.shape)

    # 找到 peak_names 在 cell_peak_data_df 列中的索引
    column_indices = [cell_peak_df_columns.index(peak) for peak in peak_names]
    print("len(column_indices):", len(column_indices))

    '''
'''
    # 使用 cell_indexs 和 column_indices，将 cell_peak_data_new 插入到扩展矩阵中
    for j, row_index in enumerate(cell_indexs):
        extended_matrix[row_index, column_indices] = cell_peak_data_new[j, :]
    '''    
'''
    for i, row_index in enumerate(cell_indexs):
        # 检查 cell_peak_data_new[i, :] 中的值是否为 NaN
        for col_idx, col in zip(column_indices, cell_peak_data_new[i, :]):
            if not np.isnan(col):  # 如果 col 不是 NaN，则更新 extended_matrix 中对应的值
                extended_matrix[row_index, col_idx] = col
                
                

    # extended_matrix=utils.standard(extended_matrix)
    # 将扩展后的矩阵转换回 DataFrame，保持行名和列名
    print("extended_matrix.shape:",extended_matrix.shape)
    extended_df = pd.DataFrame(extended_matrix, index=cell_peak_df_index, columns=cell_peak_df_columns)
    print("extended_df.shape:", extended_df.shape)

    cell_peak_data = extended_df.T
    cell_peak_data_df = cell_peak_data.T

    # 输出扩展后的矩阵
    # print(extended_df)
    print("cell_peak_data.shape:", cell_peak_data.shape)

    # cell_peak_data=utils.standard(extended_df)

    # Arc.updateData(cell_peak_data, peak_names, cell_indexs)
    #Arc.saveData(i + 1)
    file_path = os.path.join(exp_dirs, f'epoch-{i+1}-datanew')
    with open(file_path, 'wb') as f:
        pickle.dump(cell_peak_data, f)
'''


