import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
import pickle
import os

import plots
import utils
from module.Arc import Architecture
from module.GraphAE import calculateKNNgraphDistanceMatrixStatsSingleThread


class FeatureAE(Architecture):
    def __init__(self,inData,epoch):
        # inData is cell_peak or cell_ccan matrix
        self.inData=inData
        self.epoch=epoch

        self.model=AE(dim_in=self.inData.shape[1]).to(self.param['device'])
        self.lr=1e-3
        self.optimizer=optim.Adam(self.model.parameters(), self.lr)
        self.loss=nn.MSELoss(reduction='sum')

        self.loss_ls=[]
        self.tolerance=0

    def run(self,):
        X_orig, X_embed, X_recon=self.train()
        result=(X_embed.detach().cpu().numpy(), X_recon.detach().cpu().numpy())
        
        self.save(result)
        return result

    def toCluster(self,):
        X_orig, X_embed, X_recon=self.train()
        result=[X_embed.detach().cpu().numpy(), X_recon.detach().cpu().numpy()]
        result.append(calculateKNNgraphDistanceMatrixStatsSingleThread(result[0]))
        
        self.save(result)
        return result

    def save(self,data):
        filePath=os.path.join(self.exp_dirs,f"epoch{self.epoch}-featureAE")
        with open(filePath,'wb+') as f:
            pickle.dump(data,f)

    def getDataloader(self,batch_size=12800):
        dataset = utils.ExpressionDataset(self.inData)
        result = DataLoader(dataset, batch_size=batch_size, shuffle=True, **self.param['kwargs'])
        return result

    def train(self):
        train_loader=self.getDataloader()
        self.model.train()
        for epoch in range(self.param['featureAE_epoch']):
            train_loss = 0
            # 这里batchsize太大了，一个batch就完了
            # data is Tensor of shape [batch * peak]
            for batch_idx, (data, dataindex) in enumerate(train_loader): 
                data = data.type(torch.FloatTensor).to(self.param['device'])
                z, recon_batch = self.model.forward(data) 
                Loss = self.loss(recon_batch, data)

                self.optimizer.zero_grad()
                Loss.backward()
                self.optimizer.step()
                train_loss += Loss.item()
                ## Grow recon_batch, data, z at each epoch, while printing train loss
                if batch_idx == 0:
                    recon_batch_all = recon_batch
                    data_all = data
                    z_all = z
                else:
                    recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
                    data_all = torch.cat((data_all, data), 0)
                    z_all = torch.cat((z_all, z), 0)

            avg_loss=train_loss / len(train_loader.dataset)
            self.loss_ls.append(avg_loss)
            # print(f'Epoch: {epoch+1} Average loss: {avg_loss:.4f}  single sample loss: {avg_loss/data.shape[0]}')

            if self.checkStop():
                break
        plots.plot_loss(self.loss_ls,'FAE',self.epoch,Architecture.exp_dirs)
        return  data_all, z_all, recon_batch_all

    def checkStop(self):
        # todo: 这个策略还可以再改进一下
        # 当tolerance达到阈值之后并不结束，而是清零，学习率减少，知道学习率降到最低
        if len(self.loss_ls)>1 and self.loss_ls[-1]>=self.loss_ls[-2]:
            self.tolerance+=1
        if self.tolerance>50:
            print('early stop!')
            return True
        return False


class AE(nn.Module):
    ''' 
    Autoencoder for dimensional reduction
    Args:
        x: Tensor, mini-batch
        dim: int, feature dimension
    Return:
        self.decode(z): reconstructed input 
        z: feature encoding
    '''
    def __init__(self, dim_in,dim_emb=128):
        super(AE, self).__init__()
        self.dim_in = dim_in
        self.fc1 = nn.Linear(dim_in, 512)
        self.fc2 = nn.Linear(512, dim_emb)

        self.fc3 = nn.Linear(dim_emb, 512)
        self.fc4 = nn.Linear(512, dim_in)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.relu(self.fc4(h3))

        #h3 = torch.sigmoid(self.fc3(z))
        #return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim_in))
        return z, self.decode(z)