import matplotlib.pyplot as plt
import os
import seaborn as sns
from module.Arc import Architecture

def plot_loss(loss_ls,name,epoch,exp_dirs):
    file_name=f"epoch-{epoch}-loss-{name}.png"
    file_path=os.path.join(exp_dirs,file_name)

    f, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel("EPOCH")
    ax.set_ylabel("LOSS")
    ax.plot(loss_ls)
    f.savefig(file_path,format='png')
    # 这个47的选择就有问题了，然而令人惊讶的是anjunma1和haocheng是等长的
    print(f"![](../{file_path[47:]})")


def heatmap(matrix,name,epoch,exp_dirs):
    file_name=f"epoch-{epoch}-heatmap-{name}.png"
    file_path=os.path.join(exp_dirs,file_name)

    ax=sns.heatmap(matrix)
    ax.figure.savefig(file_path)


if __name__ == '__main__':
    # 调试该文件
    data_path='/nvme2/wuqiuqin/compare/cellline/count_matrix_filter2.csv'
    label_path='/nvme2/wuqiuqin/compare/cellline/metadata_filter.tsv'
    exp_dirs="/nvme2/wuqiuqin/compare/cellline2/results/test/"
    class Args():
        def __init__(self):
            pass
    args=Args()
    args.name="exp0"
    args.cutoff=0
    args.ccancutoff=0
    Arc=Architecture(data_path,label_path,exp_dirs,args)
    cell_peak_data=Architecture.data['expr']

    heatmap(cell_peak_data,"cell-peak",0,exp_dirs)
