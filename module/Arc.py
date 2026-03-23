import torch
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import numpy as np
import utils
import pickle
import os
import random
from utils import set_random_seed

class Architecture():
    # type of data: dict
    data=None
    dataUpdated=None
    # 默认的expDirs是用来调试的路径
    #expDirs="/fs/ess/PCON0022/wangqi/WQQ/CCAN/code/Buenrostro"
    cuda = torch.cuda.is_available()
    param={
        'total_epoch':100,
        'featureAE_epoch':5000,
        'graph_AE_epoch':5000,
        'ccan_cutoff':0,
        'overlap':0.5,
        'device':torch.device("cuda" if cuda else "cpu"),
        'kwargs': {'num_workers': 1, 'pin_memory': True} if cuda else {}
        }

    def print_device_info():
        """打印是否使用GPU的设备信息"""
        print(f"CUDA available: {Architecture.cuda}")
        print(f"Using device: {Architecture.param['device']}")

    def __init__(self,data_path,label_path,exp_dirs,args,seed):
        # 设置全局随机种子
        #seed = 500  # 可自定义，如42
        self.seed = seed
        set_random_seed(seed)
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        Architecture.data,Architecture.metadata=self.loadData(data_path,label_path)
        Architecture.data=utils.findTopFeatures(Architecture.data,args.cutoff)
        Architecture.dataUpdated=Architecture.data
        Architecture.exp_dirs=exp_dirs
        Architecture.param['ccan_cutoff']=args.ccancutoff

        self.true_labels=Architecture.data['true_labels']
        self.pre_labels_old=None


    def evalClusterBetween(self,pre_labels):
        ##type: np.array
        # 这里计算的时候也要考虑到长度不一致的情况！！
        if self.pre_labels_old is not None:
            ARI_between_epochs=adjusted_rand_score(self.pre_labels_old,pre_labels)
            print(f'ARI between epochs: {ARI_between_epochs}')
        self.pre_labels_old=pre_labels


    def loadData(self,dataset_path,label_path):
        print('Loading benchmarking ATAC data')
        df=pd.read_csv(dataset_path,index_col=0).T
        features=df.to_numpy()
        true_labels, metadata = self.loadTrueLabels(label_path)  # 修改 loadTrueLabels 返回 metadata
        data = {
            # cell*peak matrix
            'expr': features.astype(float),
            'peak_names':list(df.columns),
            # type: np.array
            'true_labels':self.loadTrueLabels(label_path)
            }
        print(f"sc data has {data['expr'].shape[0]} cells, {data['expr'].shape[1]} peaks")
        return data,metadata # cell*peak

    def loadTrueLabels(self,label_path):
        df=pd.read_csv(label_path,sep='\t')
        label_names=list(set(df['label']))
        print("total classes:",len(label_names))
        true_labels=[]
        for i in df['label']:
            true_labels.append(label_names.index(i))
        return np.array(true_labels),df

    def updateData(self,new_matrix,peak_names,cell_indexs):
        # 不对，这里重构矩阵之后cell的顺序也变了，会影响ARI的计算！
        print(f"Type of self.true_labels: {type(self.true_labels)}")
        print(f"Type of cell_indexs: {type(cell_indexs)}")
        print(f"cell_indexs: {cell_indexs}")
        data = {
            'expr': new_matrix,
            'peak_names':peak_names,
            'true_labels':self.true_labels[cell_indexs]
            }
        Architecture.dataUpdated=data

        print(f"Updated：matrix shape: {Architecture.dataUpdated['expr'].shape}")
        print(f"Updated：length of peak: {len(Architecture.dataUpdated['peak_names'])}")
        print(f"Updated：length of true_lables: {len(Architecture.dataUpdated['true_labels'])}")

        if self.pre_labels_old is not None:
            self.pre_labels_old=self.pre_labels_old[cell_indexs]
            print("Updated: length of pre_labels_old: ", len(self.pre_labels_old))

    def saveData(self,epoch):
        file_path=os.path.join(Architecture.exp_dirs,f'epoch-{epoch}-dataUpdated')
        with open(file_path,'wb') as f:
            pickle.dump(Architecture.dataUpdated,f)

    def getData(self):
        return Architecture.data
    def getMetadata(self):
        return Architecture.metadata
