import os
import pickle
import torch
from module.GCN import GCN
import pandas as pd 
#from concurrent.futures import ThreadPoolExecutor, as_completed

#print("CUDA available:", torch.cuda.is_available())
#print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filePath = os.path.join(directory, f"epoch{t+1}-Cluster")
with open(filePath, 'rb') as f:
    data = pickle.load(f)

cluster = data[1]



directory = "/nvme2/wuqiuqin/compare/cellline2/"
directory2 = "/nvme2/wuqiuqin/compare/cellline2/GCN/"
#directory3 = "/nvme2/wuqiuqin/compare/GM12878_HL602/results2/GM12878_HL60_seed4500/cw0.6_m2.0_h1128_h264/"
t = 0
seed = 4500


for i in range(len(cluster)):
    
    distances_path = os.path.join(directory, f"{i}_distances.pkl")
    windows_path = os.path.join(directory, f"{i}_windows.pkl")
    
    with open(distances_path, "rb") as f:
        distances = pickle.load(f)
    with open(windows_path, "rb") as f:
        windows = pickle.load(f)
        
        
    filtered_windows, filtered_distances = zip(*[(w, d) for w, d in zip(windows, distances) if len(w) > 30])
    
    # 转换为列表
    filtered_windows = list(filtered_windows)
    filtered_distances = list(filtered_distances)
    windows = filtered_windows
    distances = filtered_distances
    
    print(f"读取 {i}_distances.pkl 和 {i}_windows.pkl，窗口数: {len(windows)}, 距离矩阵数: {len(distances)}", flush=True)

    results = []  # 用于保存每个 (distance, window) 的计算结果
    metrics = []  #保存指标：roc_score, ap_score
    
    # 逐个处理每个距离和窗口对
    for j in range(len(distances)):
    #for j in range(10):  

        distance = distances[j]
        window = windows[j]

        try:
            # 处理每一对 (distance, window)
            gcn = GCN(window, directory,directory2,t+1,seed,distance,margin=2.0,p=1,q=1,h=1,g=1, lr=1e-3,hidden1_dim=64,hidden2_dim=32,device=device)
            result = gcn.run()
            results.append(result)  # 将结果添加到结果列表

            print("len(results):",len(results))
            
            roc = result[4]
            ap = result[5]
            metrics.append({
                    "index": j,
                    "roc_score": roc,
                    "ap_score": ap
                })
            
        except Exception as e:
            import traceback
            print(f"Error in processing distances[{j}] and windows[{j}]: {e}")
            traceback.print_exc()
            raise

    # 保存结果
    print(f"最终 len(results): {len(results)}, len(metrics): {len(metrics)}")
    results_path = os.path.join(directory2, f"results_{i}.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
        
    print(f"已保存 {i} 的 GCN 计算结果到 {results_path}")   
    
    metrics_path = os.path.join(directory2, f"metrics_{i}.csv")
    df = pd.DataFrame(metrics)
    print("df.shape:",df.shape)
    df.to_csv(metrics_path, index=False)
    print(f"已保存 {i} 的 roc_score 和 ap_score 到 {metrics_path}")
