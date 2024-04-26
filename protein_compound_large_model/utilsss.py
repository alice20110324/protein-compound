import torch
import os
import pandas as pd
from torch_geometric.data import Data, Batch, Dataset

class TestbedDataset(Dataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.xd=xd
        self.xt=xt
        self.y=y
        self.smile_graph=smile_graph
        self.transform=transform
        self.pre_transform=pre_transform
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
    def process(self, xd, xt, y, smile_graph):
        

        data_list = []

        for i in range(len(self.xd)):
            smiles = self.xd[i]
            target = self.xt[i]
            labels = self.y[i]

            c_size, features, edge_index = self.smile_graph[smiles]

            GCNData = Data(x=torch.Tensor(features),
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            y=torch.FloatTensor([labels]))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done.')

        # Save data_list as a .pt file
        torch.save(data_list, os.path.join(self.processed_dir, 'two_train.pt'))

        # Save preprocessed data
        data_smiles_series = pd.Series(self.xd)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, 'two_train_smiles.csv'), index=False, header=False)

        data_sequence_series = pd.Series(self.xt)
        data_sequence_series.to_csv(os.path.join(self.processed_dir, 'two_train_sequence.csv'), index=False, header=None)