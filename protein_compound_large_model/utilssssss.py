import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader, Dataset
from torch_geometric import data as DATA
import torch
import pandas as pd
class TestbedDataset(Dataset):
#class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        print('self.dataset:',self.dataset)
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
            print('self.data:',len(self.data))
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
            print('self.data:',len(self.data))

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def len(self):
        # 返回数据集中图的数量
        return len(self.processed_file_names)

    def get(self, idx):
        # 加载处理后的图数据
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        smiles_list=[]
        sequence_list=[]
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            #GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)
            smiles_list.append(xd[i])
            sequence_list.append(xt[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        
        if self.dataset=='davis_train':
            data_smiles_series = pd.Series(smiles_list)
            data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'davis_train_smiles.csv'), index=False,
                                  header=False)
        
            data_sequence_series = pd.Series(sequence_list)
            data_sequence_series.to_csv(os.path.join(self.processed_dir,
                                               'davis_train_sequence.csv'), index=False,
                                  header=None)
        if self.dataset=='davis_test':
            data_smiles_series = pd.Series(smiles_list)
            data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'davis_test_smiles.csv'), index=False,
                                  header=False)
        
            data_sequence_series = pd.Series(sequence_list)
            data_sequence_series.to_csv(os.path.join(self.processed_dir,
                                               'davis_test_sequence.csv'), index=False,
                                  header=None)
        if self.dataset=='kiba_train':
            data_smiles_series = pd.Series(smiles_list)
            data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'kiba_train_smiles.csv'), index=False,
                                  header=False)
        
            data_sequence_series = pd.Series(sequence_list)
            data_sequence_series.to_csv(os.path.join(self.processed_dir,
                                               'kiba_train_sequence.csv'), index=False,
                                  header=None)
        if self.dataset=='kiba_test':
            data_smiles_series = pd.Series(smiles_list)
            data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'kiba_test_smiles.csv'), index=False,
                                  header=False)
        
            data_sequence_series = pd.Series(sequence_list)
            data_sequence_series.to_csv(os.path.join(self.processed_dir,
                                               'kiba_test_sequence.csv'), index=False,
                                  header=None)
            
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci