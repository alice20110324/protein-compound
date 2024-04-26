#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
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
parser.add_argument('--batch_size', type=int, default=8,
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
import torch.nn.functional as F
num_epochs=200

def train(model_flag,teacher_model, student_model,trainloader,teacher_model_optimizer,student_model_optimizer):
    for epoch in range(num_epochs):
        student_model.train()
        teacher_model.eval()
        #for i, (inputs, labels) in enumerate(trainloader):
        for i, inputs in enumerate(trainloader):
            if model_flag=='seq':
                seq_data_list=[]
                seq=inputs
                lenth=len(seq)
        
                for m , s in enumerate(seq):
                    #m.to(args.device)
                    #s.to(args.device)
                    #torch.tensor(my_integer, dtype=torch.float32, device=cuda_device)
                    seq_data_list.append((str(m),s))
                #seq_data_tensor=torch.tensor(seq_data_list)
                #seq_data_tensor.to(device)
                #seq_data_list.to(args.device)
                teacher_model_optimizer.zero_grad()
                student_model_optimizer.zero_grad()
                teacher_outputs = teacher_model(seq_data_list)
                #student_outputs = student_model(seq_data_list)
            else:
                inputs.cuda()
                teacher_model_optimizer.zero_grad()
                student_model_optimizer.zero_grad()
                teacher_outputs = teacher_model(inputs)
            #inputs.to(args.device)
            #inputs.cuda()
            student_outputs = student_model(inputs)

            # 标准的交叉熵损失
            #loss_ce = criterion(student_outputs, labels)

            # 教师机与学生机输出的损失（比如使用均方误差）
            loss_kd = F.mse_loss(student_outputs.detach(), teacher_outputs.detach())
            #loss_kd = F.mse_loss(student_outputs, teacher_outputs)
            # 组合两种损失
            #loss = loss_ce + alpha * loss_kd  # alpha 是一个超参数，用于平衡两种损失
            loss = loss_kd 
            if epoch % save_interval == 0:
                # 保存模型参数到文件
                torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(epoch))
            print("train_loss:{0} at {1} epoch.".format(loss,epoch))
                
            # 计算损失并进行反向传播
            loss.backward()
            optimizer.step()


# In[2]:


davis_pd=pd.read_csv('dataset/pretrain_dataset/davis_train.csv',sep=',')
kiba_pd=pd.read_csv('dataset/pretrain_dataset/kiba_train.csv',sep=',')

two_pd = pd.concat([davis_pd, kiba_pd])
#print('two_pd:',two_pd)
print('type:',type(two_pd))
two_pd=pd.DataFrame(two_pd)
print('type:',type(two_pd))
two_pd_new=two_pd.drop_duplicates()
print('type:',type(two_pd_new))
two_pd_new.to_csv('dataset/pretrain_dataset/two_train.csv')



from loader1 import MoleculeDataset,MoleculeDatasetBig, SeqDataset,SeqMolDataset,SmileDataset
#from utilssssss import *
smilesList_train=two_pd['compound_iso_smiles']
sequenceList_train=two_pd['target_sequence']





# In[3]:


davis_pd=pd.read_csv('dataset/pretrain_dataset/davis_test.csv',sep=',')
kiba_pd=pd.read_csv('dataset/pretrain_dataset/kiba_test.csv',sep=',')

two_pd = pd.concat([davis_pd, kiba_pd])
#print('two_pd:',two_pd)
print('type:',type(two_pd))
two_pd=pd.DataFrame(two_pd)
print('type:',type(two_pd))
two_pd_new=two_pd.drop_duplicates()
print('type:',type(two_pd_new))
two_pd_new.to_csv('dataset/pretrain_dataset/two_test.csv')



from loader1 import MoleculeDataset,MoleculeDatasetBig, SeqDataset,SeqMolDataset,SmileDataset
#from utilssssss import *
smilesList_test=two_pd_new['compound_iso_smiles']
sequenceList_test=two_pd_new['target_sequence']


# In[4]:


import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utilsss import *

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

'''
all_prots = []
datasets = ['kiba','davis','pretrain_dataset']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'dataset' + dataset + '/'
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
for dt_name in ['kiba','davis','two']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('dataset/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)###########去重
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g
'''
import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
#from utilssssss import *
from utilsss import *

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
'''
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
'''    
    
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []
for dt_name in ['kiba','davis','two']:
    opts = ['train','test']
    for opt in opts:
        df = pd.read_csv('dataset/pretrain_dataset/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound_iso_smiles'] )
compound_iso_smiles = set(compound_iso_smiles)###########去重
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g


# In[5]:


#datasets = ['davis','kiba']
g_dataset='two'
processed_data_file_train = 'dataset/pretrain_dataset/processed/' + g_dataset + '_train.pt'
processed_data_file_test = 'dataset/pretrain_dataset/processed/' + g_dataset + '_test.pt'
    
df = pd.read_csv('dataset/pretrain_dataset/' + g_dataset + '_train.csv')
train_drugs, train_prots,  train_Y = list(df['compound_iso_smiles']),list(df['target_sequence']),list(df['affinity'])
#XT = [seq_cat(t) for t in train_prots]#####################
train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(train_prots), np.asarray(train_Y)


# make data PyTorch Geometric ready
print('preparing ', g_dataset + '_train.pt in pytorch format!')
gnn_train_dataset = TestbedDataset(root='dataset/pretrain_dataset/', dataset=g_dataset+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
seq_train_dataset=SeqDataset('dataset/pretrain_dataset/processed/'+g_dataset+'_train_sequence.csv')
smiles_train_dataset=SmileDataset('dataset/pretrain_dataset/processed/'+g_dataset+'_train_smiles.csv')


# In[6]:


import torch_geometric

gnn_train_dataloader=torch_geometric.data.DataLoader(gnn_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
seq_train_dataloader=torch.utils.data.DataLoader(seq_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
smiles_train_dataloader=torch.utils.data.DataLoader(smiles_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)


# In[7]:


from ModelDistillation import *
import torch
import torch.optim as optim
def main():
    seq_embd_dim=1280
    output_embd_dim=256
    teacher_model=SeqTeacher(480,256,0.5)
    teacher_model.cuda()
    
    student_model=SeqMLPStudent(1500,128,256)
    student_model.cuda()
    # 创建教师模型的优化器
    teacher_model_optimizer = optim.Adam(teacher_model.parameters(), lr=0.01)

    # 创建学生模型的优化器
    student_model_optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9)
    train('seq',teacher_model, student_model,seq_train_dataloader,teacher_model_optimizer,student_model_optimizer)
    

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# In[8]:


if __name__ == '__main__':
    
    main()


# In[ ]:




