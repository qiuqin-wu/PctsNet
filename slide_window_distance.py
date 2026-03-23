import pandas as pd
import numpy as np
import pickle
import os
from joblib import Parallel, delayed

path = "/nvme2/wuqiuqin/compare/cellline2/combined_matrices.pkl"
with open(path, "rb") as f:
    loaded_matrices = pickle.load(f)


def extract_peak_windows_and_distances(inData, window_size=500000, step_size=250000,min_rows=10,s=0.75):
    """
    使用固定步长滑动窗口提取 500kb 以内的 Peak 子矩阵，步长为 250kb。

    参数：
    - inData: 原始 DataFrame，索引格式 'chr:start-end'
    - window_size: 滑动窗口大小（默认500kb）
    - step_size: 滑动步长（默认250kb）

    返回：
    - windows_list: 窗口内的子矩阵列表
    - distances_list: 每个窗口的距离矩阵
    """
    
    # **解析行名，提取染色体 & 起始位置**
    peak_info = pd.DataFrame(inData.index.str.extract(r'(chr[\dXY]+):(\d+)-\d+'))
    peak_info.columns = ['chr', 'start']
    peak_info['start'] = peak_info['start'].astype(int)  # 转换为整数
    peak_info['index'] = inData.index  # 保存原索引
    
    # **按染色体分组**
    windows_list = []
    distances_list = []

    for chrom, group in peak_info.groupby('chr'):
        # **按起始位置排序**
        group = group.sort_values(by='start').reset_index(drop=True)
        starts = group['start'].values
        
        if len(starts) == 0:
            continue  # 跳过空染色体
        
        min_start = starts[0]  # 该染色体上的最小起始位置
        max_start = starts[-1]  # 该染色体上的最大起始位置
        
        # **按照固定步长滑动窗口**
        for start_pos in range(min_start, max_start, step_size):
            # **筛选窗口内的 peaks**
            window_mask = (group['start'] >= start_pos) & (group['start'] <= start_pos + window_size)
            window_indices = group.loc[window_mask, 'index']
            
            if len(window_indices) >= min_rows: #只有行数>=min_rows才保留
                window_data = inData.loc[window_indices]
                windows_list.append(window_data)

                # **计算距离矩阵**
                window_starts = group.loc[window_mask, 'start'].values
                dist_matrix = np.abs(window_starts[:, None] - window_starts[None, :])  # 计算距离
                # **计算 d^(-s) 形式的距离**
                with np.errstate(divide='ignore', invalid='ignore'):  # 处理除零问题
                    dist_matrix = np.power(dist_matrix, -s)
                    np.fill_diagonal(dist_matrix, 1.0)  # 避免对角线出现 inf 或 NaN
                distances_list.append(dist_matrix)

    return windows_list, distances_list


# **并行处理每个 DataFrame 并分别保存结果**
def process_and_save_matrices(loaded_matrices, save_dir, n_jobs=4,s=0.75):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key, value in loaded_matrices.items():
        if isinstance(value, pd.DataFrame):
            print(f"处理数据集 {key} ...")

            # **并行计算窗口和距离**
            windows, distances = extract_peak_windows_and_distances(value,s=s)

            # **保存结果**
            save_path_windows = os.path.join(save_dir, f"{key}_windows.pkl")
            save_path_distances = os.path.join(save_dir, f"{key}_distances.pkl")

            with open(save_path_windows, "wb") as f:
                pickle.dump(windows, f)
            with open(save_path_distances, "wb") as f:
                pickle.dump(distances, f)

            print(f"已保存 {key} 的窗口到 {save_path_windows}")
            print(f"已保存 {key} 的距离矩阵到 {save_path_distances}")


# **执行**
save_dir = "/nvme2/wuqiuqin/compare/cellline2/"
process_and_save_matrices(loaded_matrices, save_dir, n_jobs=4,s=0.75)
