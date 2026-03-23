import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
import pickle
from sklearn.preprocessing import StandardScaler

expDirs = "/nvme2/wuqiuqin/compare/cellline2/results2/cellline_seed4500/cw0.6_m2.0_h1128_h264/"
epoch = 1
filePath = os.path.join(expDirs, f"epoch{epoch}-graphAE")
with open(filePath, 'rb') as f:
    embedding = pickle.load(f)
# emb2 = data[0]

filePath2 = os.path.join(expDirs, f"epoch{epoch}-Cluster")
with open(filePath2, 'rb') as f:
    cluster = pickle.load(f)
# cluster_labels = cluster[0]
# cluster_lists_of_idx = cluster[1]
# len(cluster_labels)


data_path = '/nvme2/wuqiuqin/compare/cellline/count_matrix_filter2.csv'
label_path = '/nvme2/wuqiuqin/compare/cellline/metadata_filter.tsv'

cell_peak_data = pd.read_csv(data_path, index_col=0)

row_names = cell_peak_data.index
col_names = cell_peak_data.columns
# 将 cell_peak_data 转换为稀疏矩阵
sparse_cell_peak_data = csr_matrix(cell_peak_data.values)


def create_type_matrix(sparse_cell_peak_data, cluster, embedding):
    cell_types = cluster[0]
    cluster_cells = cluster[1]
    embeddings = embedding[0]

    # 优化：预计算非零细胞索引
    non_zero_cells = (sparse_cell_peak_data != 0).toarray()  # 计算哪些cell在peak下非零 (将稀疏矩阵转为密集数组)

    type_matrices = {}

    # 遍历每个细胞类型
    for cell_type_idx, cell_indices in enumerate(cluster_cells):
        # 创建一个零矩阵，用于保存每个细胞类型的结果，大小为 (157283, 16)
        type_matrix = np.zeros((sparse_cell_peak_data.shape[0], embeddings.shape[1]))

        # 提前过滤出该细胞类型的嵌入
        embeddings_for_type = embeddings[cell_indices, :]

        # 对于每个peak
        for peak_index in range(sparse_cell_peak_data.shape[0]):
            # 获取该 peak 对应的 cell 的非零值
            peak_data = sparse_cell_peak_data[peak_index, :].toarray().flatten()

            # 找到该细胞类型下非零的cell索引
            valid_cells = np.where(non_zero_cells[peak_index, cell_indices])[0]

            if len(valid_cells) > 0:
                # 直接提取有效细胞的嵌入向量
                embeddings_for_cells = embeddings_for_type[valid_cells, :]
                # 对这些细胞的嵌入向量进行求均值，结果是一个 16 维向量
                type_matrix[peak_index, :] = np.mean(embeddings_for_cells, axis=0)
        print("type_matrix.shape:", type_matrix.shape)
        # 将每个细胞类型的矩阵保存到字典中
        type_matrices[cell_type_idx] = pd.DataFrame(type_matrix,index=cell_peak_data.index)

    return type_matrices




# 调用函数
type_matrices = create_type_matrix(sparse_cell_peak_data, cluster, embedding)

print("type(type_matrices):", type(type_matrices))
print("len(type_matrices):", len(type_matrices))



peak_embedding_path = "/nvme2/wuqiuqin/compare/cellline/peak_embedding.csv"

peak_embedding = pd.read_csv(peak_embedding_path, index_col=0)  # 128维

# 选择标准化方法（Z-score 标准化）
scaler = StandardScaler()

peak_embedding_scaled = pd.DataFrame(scaler.fit_transform(peak_embedding),
                                     index=peak_embedding.index,
                                     columns=peak_embedding.columns)

# 存储拼接后的 DataFrame
combined_matrices = {}

for cell_type, matrix in type_matrices.items():
    matrix_scaled = pd.DataFrame(scaler.fit_transform(matrix),
                                 index=matrix.index,
                                 columns=matrix.columns)  # 保留行名和列名
    combined_matrices[cell_type] = pd.concat([matrix_scaled, peak_embedding_scaled], axis=1)

print("type(combined_matrices):", type(combined_matrices))
print("len(combined_matrices):", len(combined_matrices))

save_dir = "/nvme2/wuqiuqin/compare/cellline2/"
save_path = os.path.join(save_dir, "combined_matrices.pkl")

with open(save_path, "wb") as f:
    pickle.dump(combined_matrices, f)
