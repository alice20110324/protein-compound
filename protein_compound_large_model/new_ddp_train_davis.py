#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import roc_auc_score
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from loader1 import MoleculeDataset,MoleculeDatasetBig, SeqDataset,SeqMolDataset,SmileDataset#########################
import torch
import torch
#import args
from torchvision.models import resnet18

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

#from loader import MoleculeDataset#################
#from torch_geometric.data import DataLoader
#from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred,GNN_graphpred_1


from splitters import scaffold_split,scaffold_split_1
import pandas as pd
import os
import shutil
from tensorboardX import SummaryWriter
#import esm2_t33_650M_UR50D
import esm
import time
import torch
from torch import nn
import argparse
import torch.nn.functional as F
import os
import numpy as np
import random
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
#from finetune.tokenizer.tokenizer import MolTranBertTokenizer
from fast_transformers.masking import LengthMask as LM
#from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
from apex import optimizers
import subprocess
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
#from utils import normalize_smiles
import sys
sys.path.append('finetune/')
from utilss import normalize_smiles
from tokenizer.tokenizer import MolTranBertTokenizer
from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
#from SeqMolModel import InteractionModel,InteractionModel_1,SequenceModel,InteractionModel_4
#from SeqMolSmile import InteractionModel_4
#from SeqMolModel import InteractionModel_4
from SeqMolSmile_model2 import InteractionModel_4
#print(torch.cuda.is_available())
import torch
torch.cuda.current_device()
torch.cuda._initialized = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# Training settings
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int,default=0,
                        help='which gpu to use if any (default: 0)')#0000
parser.add_argument('--gpu',default='0,1,2')
parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.01)')
parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
parser.add_argument('--gnn_type', type=str, default="gin")
parser.add_argument('--dataset', type=str, default = 'davis', help='root directory of dataset. For now, only classification.')
#parser.add_argument('--input_model_file', type=str, default = 'None', help='filename to read the model (if there is any)')
parser.add_argument('--input_model_file', type=str, default = 'Mole-BERT', help='filename to read the model (if there is any)')
parser.add_argument('--filename', type=str, default = '', help='output filename')
parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
#parser.add_argument('--gpu', type=int, default=0, help='')
parser.add_argument('--rank',type=int,default=0,help='')
parser.add_argument('--world_size', type=float,default=0.1,help='')
parser.add_argument('--dist_backend ',type=str, default='nccl',help='')

parser.add_argument('--n_head', type=int, default = 12, help='number of workers for dataset loading')
parser.add_argument('--local_rank',type=int,default=0,help='')
parser.add_argument('--n_layer', type=int, default = 12, help='number of workers for dataset loading')
parser.add_argument('--d_dropout', type=float, default = 0.1, help='number of workers for dataset loading')
parser.add_argument('--n_embd', type=int, default = 768, help='number of workers for dataset loading')
parser.add_argument('--dropout', type=float, default = 0.1, help='number of workers for dataset loading')
parser.add_argument('--lr_start', type=float, default =  3e-5, help='number of workers for dataset loading')
parser.add_argument('--max_epochs', type=int, default = 500, help='number of workers for dataset loading')
parser.add_argument('--num_feats', type=int, default = 32, help='number of workers for dataset loading')
parser.add_argument('--checkpoint_every', type=int, default = 100, help='number of workers for dataset loading')
parser.add_argument('--seed_path', type=str, default =  'data/checkpoints/N-Step-Checkpoint_3_30000.ckpt', help='number of workers for dataset loading')
parser.add_argument('--dims', type=list, default = [ 768, 768, 768, 1], help='number of workers for dataset loading')

args = parser.parse_args(args=[])###############33

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
# 初始化分布式训练环境
'''
def init_distributed():
    print('initializing########')
    torch.cuda.set_device(args.local_rank)
    #dist.init_process_group(backend='nccl', init_method='env://')
    os.environ['MASTER_ADDR'] = 'localhost'

    os.environ['MASTER_PORT'] = '5652'
    os.environ['RANK']=str(0)
    os.environ['WORLD_SIZE']=str(4)
    print('initializing########')
    #dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 0.2)
    #dist.init_process_group(backend='nccl', init_method='file:///mnt/nfs/sharedfile',
                            #world_size=4, rank=args.rank)
    #dist.init_process_group("gloo", init_method='env://',rank=args.rank, world_size=4)
    #dist.init_process_group('nccl', init_method='env://')
    dist.init_process_group(backend='gloo',init_method='env://',rank=0,world_size=1)
    print('initializing########')
    #————————————————
    #版权声明：本文为CSDN博主「软软没吃饱」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    #原文链接：https://blog.csdn.net/weixin_57634679/article/details/129082198

# 定义模型
init_distributed()
'''
def prepare():
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '19198'  
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
def init_ddp(local_rank):
    
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    print('local_rank:',local_rank)
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g



def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


class LightningModule(pl.LightningModule):

    def __init__(self, config, tokenizer):
        super(LightningModule, self).__init__()

        self.config = config
        #self.hparams = config
        #self.mode = config.mode
        self.save_hyperparameters(config)
        self.tokenizer=tokenizer
        '''
        self.min_loss = {
            self.hparams.measure_name + "min_valid_loss": torch.finfo(torch.float32).max,
            self.hparams.measure_name + "min_epoch": 0,
        }
        '''
        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), config.n_embd
        # input embedding stem
        
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        #print('self.tok_emb:',self.tok_emb)
        self.drop = nn.Dropout(config.d_dropout)
        
        ## transformer
        self.blocks = builder.get()
        #self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        #self.train_config = config
        #if we are starting from scratch set seeds
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################
        '''
        self.fcs = []  
        self.loss = torch.nn.L1Loss()
        self.net = self.Net(
            config.n_embd, dims=config.dims, dropout=config.dropout,
        )
        '''


    class Net(nn.Module):
        dims = [150, 50, 50, 2]


        def __init__(self, smiles_embed_dim, dims=dims, dropout=0.2):
            super().__init__()
            self.desc_skip_connection = True 
            self.fcs = []  # nn.ModuleList()
            #print('dropout is {}'.format(dropout))

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, 1)

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)

            return z

    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def get_loss(self, smiles_emb, measures):

        z_pred = self.net.forward(smiles_emb).squeeze()
        measures = measures.float()

        return self.loss(z_pred, measures), z_pred, measures

    



def collate_fn(batch):
    print('collate_batch:',batch)
    print('collate_batch_0:',batch[0])
    tokenizer = MolTranBertTokenizer('finetune/bert_vocab.txt')    
    tokens = tokenizer.batch_encode_plus([ smile for smile in batch], padding=True, add_special_tokens=True)
    return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']))


import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset

class MultiDatasetMixin:
    def __init__(self, dataset1, dataset2,dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3=dataset3

    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2),len(self.dataset3))

    def __getitem__(self, idx):
        
        return self.dataset1[idx], self.dataset2[idx],self.dataset3[idx]

class CustomMultiDataset(MultiDatasetMixin, Dataset):###############3extends two classes
    def __init__(self, dataset1, dataset2,dataset3):
        MultiDatasetMixin.__init__(self, dataset1, dataset2,dataset3)

class SharedRandomSampler(Sampler):
    def __init__(self, data_source, generator=None):
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        return (self.data_source[i] for i in torch.randperm(n, generator=self.generator))

    def __len__(self):
        return len(self.data_source)
shared_generator = torch.Generator().manual_seed(42)
#chartGPT太厉害了，杀了我吧
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader

class MultiDataLoader:
    def __init__(self, dataloader1, dataloader2,dataloader3):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.dataloader3=dataloader3

    def __iter__(self):
        for data1, data2,data3 in zip(self.dataloader1, self.dataloader2, self.dataloader3):
            yield data1, data2, data3

    def __len__(self):
        return min(len(self.dataloader1), len(self.dataloader2), len(self.dataloader3))

    def set_shuffle(self, shuffle):
        self.dataloader1.shuffle = shuffle
        self.dataloader2.shuffle = shuffle
        self.dataloader3.shuffle=shuffle

class DistributedMultiDataLoaders:
    def __init__(self, dataset1, dataset2, dataset3, batch_size,generator, num_workers, distributed=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.pin_memory=True
        self.generator=generator

        # Create DistributedSampler if in distributed mode
        if self.distributed:
            self.sampler1 = DistributedSampler(dataset1)
            self.sampler2 = DistributedSampler(dataset2)
            self.sampler3 = DistributedSampler(dataset3)
        else:
            self.sampler1 = None
            self.sampler2 = None
            self.sampler3 = None

        # Create DataLoader instances
        self.dataloader1 = DataLoader(
            dataset1,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=self.sampler1,
            pin_memory=self.pin_memory,
            generator=self.generator,
            shuffle=False
        )

        self.dataloader2 =GeometricDataLoader(
            dataset2,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=self.sampler2,
            pin_memory=self.pin_memory,
            generator=self.generator,
            shuffle=False
        )

        self.dataloader3 = DataLoader(
            dataset3,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=self.sampler3,
            pin_memory=self.pin_memory,
            generator=self.generator,
            shuffle=False
        )

        # Wrap DataLoader with DistributedDataParallel if in distributed mode
        
        if self.distributed:
            self.dataloader1 = DistributedDataParallel(self.dataloader1)
            self.dataloader2 = DistributedDataParallel(self.dataloader2)
            self.dataloader3 = DistributedDataParallel(self.dataloader3)
        
    def set_shuffle(self, shuffle):
        self.dataloader1.shuffle = shuffle
    def set_epoch(self, epoch):
        self.dataloader1.sampler.set_epoch(epoch)
        self.dataloader2.sampler.set_epoch(epoch)
        # 迭代所有DataLoader实例
        self.iter1 = iter(self.dataloader1)
        self.iter2 = iter(self.dataloader2)
        self.iter3 = iter(self.dataloader3)
        return self

    def __next__(self):
        # 并行获取来自每个DataLoader的数据
        try:
            batch1 = next(self.iter1)
            batch2 = next(self.iter2)
            batch3 = next(self.iter3)
        except StopIteration:
            # 重新设置采样器，以便下一个epoch可以重新开始
            self.sampler1.set_epoch(self.sampler1.epoch + 1)
            self.sampler2.set_epoch(self.sampler2.epoch + 1)
            self.sampler3.set_epoch(self.sampler3.epoch + 1)
            raise StopIteration

        return batch1, batch2, batch3

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import DataLoader as GeometricDataLoader
class SynchronizedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        #num_replicas=进程数
        self.randnum = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # 使用固定的种子和 epoch =randnum值来生成 shuffle 的索引
        g = torch.Generator()
        g.manual_seed(self.randnum)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # 将索引分割给不同的 replica
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

class MyDataLoader(DataLoader,GeometricDataLoader):
    def __init__(self, dataset1, dataset2, dataset3, batch_size, num_replicas, rank,generator):
        self.sampler1 = SynchronizedSampler(dataset1, num_replicas=num_replicas, rank=rank)
        self.sampler2 = SynchronizedSampler(dataset2, num_replicas=num_replicas, rank=rank)
        self.sampler3 = SynchronizedSampler(dataset3, num_replicas=num_replicas, rank=rank)

        self.dataloader1 = torch.util.data.DataLoader(dataset1, batch_size=batch_size, sampler=self.sampler1,pin_memory=True,generator=generator)
        self.dataloader2 = torch_geometric.data.DataLoader(dataset2, batch_size=batch_size, sampler=self.sampler2,pin_memor=True,generator=generator)
        self.dataloader3 = torch.util.data.DataLoader(dataset3, batch_size=batch_size, sampler=self.sampler3,pin_memory=True,generator=generator)

    def set_randnum(self, randnum):
        self.sampler1.set_randnum(randnum)
        self.sampler2.set_randnum(randnum)
        self.sampler3.set_randnum(randnum)

# 使用示例
# 假设 dataset1, dataset2, dataset3 是已经定义的 Dataset 实例
batch_size = 32
num_replicas = 2  # 假设有2个进程
rank = 0  # 当前进程的排名

loader = MyDataLoader(dataset1, dataset2, dataset3, batch_size, num_replicas, rank)
    
    
    
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class SynchronizedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # 使用固定的种子和 epoch 值来生成 shuffle 的索引
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # 将索引分割给不同的 replica
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

class MyDataLoader(DataLoader):
    def __init__(self, dataset1, dataset2, dataset3, batch_size, num_replicas, rank):
        self.sampler1 = SynchronizedSampler(dataset1, num_replicas=num_replicas, rank=rank)
        self.sampler2 = SynchronizedSampler(dataset2, num_replicas=num_replicas, rank=rank)
        self.sampler3 = SynchronizedSampler(dataset3, num_replicas=num_replicas, rank=rank)

        self.dataloader1 = DataLoader(dataset1, batch_size=batch_size, sampler=self.sampler1)
        self.dataloader2 = DataLoader(dataset2, batch_size=batch_size, sampler=self.sampler2)
        self.dataloader3 = DataLoader(dataset3, batch_size=batch_size, sampler=self.sampler3)

    def set_epoch(self, epoch):
        self.sampler1.set_epoch(epoch)
        self.sampler2.set_epoch(epoch)
        self.sampler3.set_epoch(epoch)
        
        
        
        
import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utilssssss import *

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  


all_prots = []
datasets = ['kiba','davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'dataset/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e ]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y","rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train','test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)  
        if opt=='train':
            rows,cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows,cols = rows[valid_fold], cols[valid_fold]
        with open('dataset/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [ drugs[rows[pair_ind]]  ]
                ls += [ prots[cols[pair_ind]]  ]
                ls += [ affinity[rows[pair_ind],cols[pair_ind]]  ]
                f.write(','.join(map(str,ls)) + '\n')       
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))
    all_prots += list(set(prots))
    
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []
for dt_name in ['kiba','davis']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('dataset/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)###########去重
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

datasets = ['davis','kiba']
# convert to PyTorch data format
g_dataset=args.dataset


#复制代码import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

processed_data_file_train = 'dataset/processed/' + g_dataset + '_train.pt'
processed_data_file_test = 'dataset/processed/' + g_dataset + '_test.pt'
    
df = pd.read_csv('dataset/' + g_dataset + '_train.csv')
train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
#XT = [seq_cat(t) for t in train_prots]#####################
train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)
df = pd.read_csv('dataset/' + g_dataset + '_test.csv')
test_drugs, test_prots,  test_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
#XT = [seq_cat(t) for t in test_prots]
test_drugs, test_prots,  test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

# make data PyTorch Geometric ready
print('preparing ', dataset + '_train.pt in pytorch format!')
gnn_train_dataset = TestbedDataset(root='dataset', dataset=g_dataset+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
seq_train_dataset=SeqDataset('dataset/processed/'+g_dataset+'_train_sequence.csv')
smiles_train_dataset=SmileDataset('dataset/processed/'+g_dataset+'_train_smiles.csv')
print('preparing ', dataset + '_test.pt in pytorch format!')
gnn_test_dataset = TestbedDataset(root='dataset', dataset=g_dataset+'_test', xd=test_drugs, xt=test_prots, y=test_Y,smile_graph=smile_graph)
seq_test_dataset=SeqDataset('dataset/processed/'+g_dataset+'_test_sequence.csv')
smiles_test_dataset=SmileDataset('dataset/processed/'+g_dataset+'_test_smiles.csv')
print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')  

# 创建并应用自定义的 Sampler
seq_train_sampler = SharedRandomSampler(seq_train_dataset, generator=shared_generator)
gnn_train_sampler = SharedRandomSampler(gnn_train_dataset, generator=shared_generator)
smiles_train_sampler = SharedRandomSampler(smiles_train_dataset, generator=shared_generator)

seq_test_sampler = SharedRandomSampler(seq_test_dataset, generator=shared_generator)
gnn_test_sampler = SharedRandomSampler(gnn_test_dataset, generator=shared_generator)
smiles_test_sampler = SharedRandomSampler(smiles_test_dataset, generator=shared_generator)



def train(args, epoch, model,  loader, optimizer,scaler):
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    device=args.device
    model.train()
    save_pt='results/davis/model1/'
    #epoch_iter = tqdm(loader, desc="Iteration")
    for step, (A,B,C) in enumerate(loader):
        print('eposh,step:',epoch,step)
        seq_data_list=[]
        seq=A
        lenth=len(seq)
        
        for m , s in enumerate(seq):
            seq_data_list.append((str(m),s))
        B=B.to(device)
        print('B%%%%%%%%%%%%:',B)
        D,E=C
        D=D.to(device)
        E=E.to(device)
        C=(D,E)
        
        pred=model(seq_data_list,B,C)#model is error
        print('pred_model:',pred)
        y_true = B.y.view(pred.shape).to(torch.float32)
        
        loss1=criterion(pred,y_true)
        
        loss=loss1
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        scaler.scale(loss).backward()  ###
        scaler.step(optimizer)  ###
        scaler.update()
def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    
    #for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    with torch.no_grad():
        for step, (A,B,C) in enumerate(loader):
            seq_data_list=[]
            seq=A
            lenth=len(seq)
            for m , s in enumerate(seq):
                seq_data_list.append((str(m),s))
            B=B.to(device)
            D,E=C
            D=D.to(device)
            E=E.to(device)##################
            C=(D,E)
        
        
            #pred=model(seq_data_list,B,C)#model is error
            
            pred=model(seq_data_list,B,C)#model is error
            y_true=B.y.view(pred.shape).to(torch.float32)
            val_loss = mse_criterion(pred, y_true)
            #print(f'Validation Loss: {val_loss.item():.4f}')    
                             

    
    return val_loss.item()


import torch, gc
protein_model, protein_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
num_tasks=1
if torch.cuda.is_available():
    protein_model.to(device)
for i,p in enumerate(protein_model.parameters()):
    p.requires_grad = False
    
protein_model.eval()  # disables dropout for deterministic results
molecular_model = GNN_graphpred_1(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
if not args.input_model_file == "None":###############
    print('Not from scratch')
    molecular_model.from_pretrained('model_gin/{}.pth'.format(args.input_model_file))
    print('rese:model_gin')
molecular_model.to(device)


print('molecular load############')
for i,p in enumerate(molecular_model.parameters()):
    p.requires_grad = False#freezing parameters
#freezing parameters
for i,p in enumerate(protein_model.parameters()):
    p.requires_grad = False#freezing parameters
print('parameter frozen###########')

    
    
margs = args
tokenizer = MolTranBertTokenizer('finetune/bert_vocab.txt')
#seed.seed_everything(margs.seed)

print('smile_model_start$$$$$')
if margs.seed_path == '':
    #print("# training from scratch")
    smile_model = LightningModule(margs, tokenizer)
else:
    #print("# loaded pre-trained model from {args.seed_path}")
    smile_model = LightningModule(margs, tokenizer).load_from_checkpoint(margs.seed_path, strict=False, config=margs, tokenizer=tokenizer, vocab=len(tokenizer.vocab))#########################33

#print('model:',smile_model)
#freezing parameters

print('smile_model_load************')
for i,p in enumerate(smile_model.parameters()):
    p.requires_grad = False
    
print('smile_model_parameters***********')


model= InteractionModel_4(protein_model=protein_model,molecular_model=molecular_model,smile_model=smile_model,protein_embd_dim=1280,num_tasks=1,device=device,mol_embd_dim=300,smile_embd_dim=768) 
model.to(device)
model = model.cuda()

#model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
# 将模型封装为DistributedDataParallel对象
 
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CombinedSampler(Sampler):
    def __init__(self, dataset, distributed_sampler, generator=None):
        self.dataset = dataset
        self.distributed_sampler = distributed_sampler
        self.generator = generator or torch.Generator()

    def __iter__(self):
        # 获取由 DistributedSampler 生成的索引
        distributed_indices = list(self.distributed_sampler)
        
        # 打乱这些索引
        shuffled_indices = torch.randperm(len(distributed_indices), generator=self.generator).tolist()
        return (distributed_indices[i] for i in shuffled_indices)

    def __len__(self):
        return len(self.distributed_sampler)

# 示例用法
dataset = CustomDataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
distributed_sampler = DistributedSampler(dataset, num_replicas=1, rank=0)
combined_sampler = CombinedSampler(dataset, distributed_sampler)

dataloader = DataLoader(dataset, sampler=combined_sampler, batch_size=2)
def main(local_rank,args):
    



    # 初始化分布式训练环境
    #init_distributed()
    #init_distributed_mode(args)
    print('distributed#########')
    # 创建模型
     ############
    init_ddp(local_rank)
    model = DistributedDataParallel(model,device_ids=[local_rank])   
    
    g=get_ddp_generator()
    #seq_mol_smile_train_multi_loader=DistributedMultiDataLoaders(seq_train_dataset, gnn_train_dataset, smiles_train_dataset, batch_size=args.batch_size,generator=g, num_workers=args.num_workers, distributed=True)
    #seq_mol_smile_train_multi_loader.set_shuffle(True)
    
    #seq_mol_smile_test_multi_loader=DistributedMultiDataLoaders(seq_test_dataset,gnn_test_dataset,smies_test_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
    #seq_mol_smile_test_multi_loader.set_shuffle(True)
    '''
    # 加载数据集
    gnn_train_sampler = DistributedSampler(gnn_train_dataset)
    seq_train_sampler = DistributedSampler(seq_train_dataset)
    smiles_train_sampler = DistributedSampler(smiles_train_dataset)
    
    
    print('sampler#############')
    gnn_test_sampler = DistributedSampler(gnn_test_dataset)
    seq_test_sampler = DistributedSampler(seq_test_dataset)
    smiles_test_sampler = DistributedSampler(smiles_test_dataset)
    g = get_ddp_generator()##########################
    seq_train_dataloader1 = torch.utils.data.DataLoader(seq_train_dataset,generator=g,sampler=seq_train_sampler, pin_memory=True,batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)######False
    mol_train_dataloader2 = GeometricDataLoader(gnn_train_dataset, generator=g,sampler=gnn_train_sampler,pin_memory=True,batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
    smile_train_dataloader3=torch.utils.data.DataLoader(smiles_train_dataset,generator=g,sampler=smiles_train_sampler,pin_memory=True,batch_size=args.batch_size,collate_fn=collate_fn, shuffle=False,num_workers=args.num_workers)
    #seq_mol_smile_train_multi_loader = MultiDataLoader(seq_train_dataloader1, mol_train_dataloader2,smile_train_dataloader3)
    # Set the shuffle parameter simultaneously for both dataloaders
    seq_mol_smile_train_multi_loader.set_shuffle(True)
        
    print('dataloader####################################')   
        
    seq_test_dataloader1 = torch.utils.data.DataLoader(seq_test_dataset,generator=g,sampler=seq_train_sampler,pin_memory=True, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)######False
    mol_test_dataloader2 = GeometricDataLoader(gnn_test_dataset,generator=g, sampler=gnn_train_sampler,pin_memory=True,batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
    smile_test_dataloader3=torch.utils.data.DataLoader(smiles_test_dataset,generator=g,sampler=smiles_train_sampler,pin_memory=True,batch_size=args.batch_size,collate_fn=collate_fn, shuffle=False,num_workers=args.num_workers)

    seq_mol_smile_test_multi_loader = MultiDataLoader(seq_test_dataloader1, mol_test_dataloader2,smile_test_dataloader3)
    # Set the shuffle parameter simultaneously for both dataloaders
    seq_mol_smile_test_multi_loader.set_shuffle(True)
    #train_dataset = ...  # 替换为您自己的训练数据集
    #train_sampler = DistributedSampler(train_dataset)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    '''
    
    seq_mol_smile_train_dataloader=MyDataLoader(seq_train_dataset, gnn_train_dataset, smile_train_dataset, batch_size=args.batch_size, num_replicas=dist.world_size, rank=dist.get_rank,generator=g)
    seq_mol_smie_train_dataloader.set_randnum(100)
    
    seq_mol_smile_test_dataloader=MyDataLoader(seq_test_dataset, gnn_test_dataset, smile_test_dataset, batch_size=args.batch_size, num_replicas=dist.world_size, rank=dist.get_rank,generator=g)
    seq_mol_smie_test_dataloader.set_randnum(100)
    
    
    
    
    
    criterion=nn.SmoothL1Loss()
    mse_criterion=nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    gc.collect()
    torch.cuda.empty_cache()

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    if not args.filename == "":
        fname = 'runs/seq_mol_finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        if local_rank == 0:  ### 
            print('begin training of epoch {}'.format((epoch + 1)/(args.epochs)))
        seq_mol_smile_train_multi_loader.set_epoch(epoch)  ### 
        train(model, seq_mol_smile_train_multi_loader, criterion, optimizer, scaler)
    if local_rank == 0:
        print('begin testing')
        eval(model, seq_mol_smile_test_multi_loader)
    
    if local_rank == 0:  ##
        train_loss=eval(model,seq_mol_smile_train_multi_loader)
        test_loss=eval(model,seq_mol_smile_test_multi_loader)
        print("train: %f  test: %f" %(train_loss, test_loss))
        if epoch %50==0:
            torch.save({
                'model': model.state_dict(),
                'scaler': scaler.state_dict()
            }, 'results/davis/model1/ddp_davis_model1_checkpoint_{}.pt'.format(epoch))
    dist.destroy_process_group()
    
    '''
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train(args, epoch, model, device, seq_mol_smile_train_multi_loader, optimizer)
        print("====Evaluation")
        if args.eval_train:
            train_loss = eval(args, model, device, seq_mol_smile_train_multi_loader)
        else:
            print("omit the training accuracy computation")
            train_loss = 0
            
        test_loss = eval(args, model, device, seq_mol_smile_test_multi_loader)
        with open(results_save_file, 'w+') as f:
            f.write(str(epoch)+'\t'+str(train_loss)+'\t'+str(val_loss)+'\n')
        
        print("train: %f  test: %f" %(train_loss, test_loss))
    
        '''
if __name__ == '__main__':
    prepare()
    time_start = time.time()
    mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count())
    time_elapsed = time.time() - time_start
    print('\ntime elapsed: {time_elapsed:.2f} seconds')
        

