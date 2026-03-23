import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import torch
import scipy.sparse as sp
from sklearn import preprocessing
import numpy as np


def flatten(t):
    return [item for sublist in t for item in sublist]

def loadTrueLabels(label_path):
    # label_path='/fs/ess/PCON0022/qiuqinwu/scATAC-seq_data/Buenrostro_2018_bulkpeaks/metadata.tsv'
    df=pd.read_csv(label_path,sep=' ')
    label_names=list(set(df['label']))
    print("total classes:",len(label_names))
    true_labels=[]
    for i in df['label']:
        true_labels.append(label_names.index(i))
    return true_labels


#select top cutoff col 
def findTopFeatures(data,cutoff):
    # cutoff取0的时候全部数据
    if cutoff!=0:
        data_matrix=data['expr']
        data_matrix = np.where(data_matrix > 1, 1, np.where(data_matrix < 1, 0, data_matrix))
        index=np.argsort(np.sum(data_matrix,0))[-cutoff:]
        data_matrix=data_matrix[:,index]
        print("After cutoff:",data_matrix.shape)
        data['expr']=data_matrix
        data['peak_names']=list(np.array(data['peak_names'])[index])
    return data


'''
#select top 80% col
def findTopFeatures(data,cutoff):
    # cutoff取0的时候全部数据
    if cutoff!=0:
        data_matrix=data['expr']
        data_matrix = np.where(data_matrix > 1, 1, np.where(data_matrix < 1, 0, data_matrix))
        num_cols = data_matrix.shape[1]
        top_80_percent = int(np.floor(num_cols * cutoff))
        index = np.argsort(np.sum(data_matrix, axis=0))[-top_80_percent:]
        #index=np.argsort(np.sum(data_matrix,0))[-cutoff:]
        data_matrix=data_matrix[:,index]
        print("After cutoff:",data_matrix.shape)
        data['expr']=data_matrix
        data['peak_names']=list(np.array(data['peak_names'])[index])
    return data
'''


class ExpressionDataset(Dataset):
    def __init__(
        self, 
        X=None, 
        transform=None
        ):
        """
        Args:
            X : ndarray (dense) or list of lists (sparse) [cell * peak]
            transform (callable, optional): apply transform function if not none
        """
        self.X = X # [cell * peak]

        # save nonzero
        # self.nz_i,self.nz_j = self.features.nonzero()
        self.transform = transform

    def __len__(self):
        return self.X.shape[0] # of cell

    def __getitem__(self, idx):
        
        # Get sample (one cell)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        # Convert to Tensor
        if type(sample) == sp.lil_matrix:
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        # Transform
        if self.transform:
            sample = self.transform(sample)

        return sample, idx



def standard(matrix):
    assert (matrix>=0).all(), "matrix中元素不应该小于0"
    # 缩放到[0,1]
    result = preprocessing.MinMaxScaler().fit_transform(matrix)
    assert (result>=0).all(), "matrix中元素不应该小于0"
    return result

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def remove_overlap(peak_ls,t):
    # 输入是排序完的peak列表
    # t是阈值
    count=len(peak_ls)
    index=0
    while index<len(peak_ls)-1:
        new_peak_ls=[i for i in peak_ls[index+1:] if jaccard_similarity(i,peak_ls[index])<t]
        peak_ls=peak_ls[:index+1]+new_peak_ls
        # print(peak_ls)
        index+=1
    
    print("total remove overlap:",count-len(peak_ls))
    return peak_ls

def set_random_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 调试该文件
    # import numpy as np
    # from sklearn.metrics import jaccard_score
    # y_true = np.array([1,2,3,4])
    # y_pred = np.array([1,2,3,6])

    # print(jaccard_similarity(y_true,y_pred))

    # peak_ls=[[1,2,3],[1,2],[1],[4,5,6],[3,2,1],[4,5,6,7]]

    # print(remove_overlap(peak_ls,0.3))
    set_random_seed(args.seed)

    pass