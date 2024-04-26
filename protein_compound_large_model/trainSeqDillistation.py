#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[17]:


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
parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=200,
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
parser.add_argument('--num_workers', type=int, default = 0, help='number of workers for dataset loading')
#parser.add_argument('--gpu', type=int, default=0, help='')
parser.add_argument('--rank',type=int,default=0,help='')
parser.add_argument('--world_size', type=float,default=0.1,help='')
parser.add_argument('--dist_backend ',type=str, default='nccl',help='')
parser.add_argument('--model_protein',type=nn.Module)
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
torch.cuda.empty_cache()

def train(teacher_model, student_model,trainloader,teacher_model_optimizer,student_model_optimizer):
    
        for i, inputs in enumerate(trainloader):
            
            teacher_model_optimizer.zero_grad()
            # 生成教师和学生模型的输出
            student_model_optimizer.zero_grad()
            teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)

            # 标准的交叉熵损失
            #loss_ce = criterion(student_outputs, labels)

            # 教师机与学生机输出的损失（比如使用均方误差）
            loss_kd = F.mse_loss(student_outputs, teacher_outputs.detach())

            # 组合两种损失
            #loss = loss_ce + alpha * loss_kd  # alpha 是一个超参数，用于平衡两种损失
            loss = loss_kd 
            '''
            if epoch % save_interval == 0:
                # 保存模型参数到文件
                torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(epoch))
            '''
            print("train_loss:{0} at {1} epoch.".format(loss,epoch))
                
            # 计算损失并进行反向传播
            loss.backward()
            optimizer.step
            
from ModelDistillation import *
import GPUtil
def  getGPU():
    # 获取所有GPU的详细信息
    gpus = GPUtil.getGPUs()

    # 打印每个GPU的信息
    for gpu in gpus:
        print(f"GPU: {gpu.name}")
        print(f"  GPU ID: {gpu.id}")
        print(f"  显存总量: {gpu.memoryTotal}MB")
        print(f"  显存使用: {gpu.memoryUsed}MB")
        print(f"  显存空闲: {gpu.memoryFree}MB")
        print(f"  GPU负载: {gpu.load*100}%")
def trainSeq(epoch,protein_model,trainloader,student_model,student_model_optimizer,protein_embd_dim):
        loss=0
        inputs_1500_list=[]
        for i, inputs in enumerate(trainloader):
            
                seq_data_list=[]
                
                lenth=len(inputs)
                #print('seq type:',type(seq)) 
                for m , s in enumerate(inputs):
                    seq_data_list.append((str(m),s))
                student_model_optimizer.zero_grad()
                
                
                #del seq
                
                #inputs.cuda()
                
                student_model_optimizer.zero_grad()
                teacher_outputs = SeqInferTeacher(protein_model,seq_data_list,protein_embd_dim)
                student_outputs = student_model(inputs)
            
                loss_kd = F.mse_loss(student_outputs.detach(), teacher_outputs.detach())
            
                loss = loss_kd 
            
                print("train_loss:{0} at{1} epoch.".format(loss,epoch))
                getGPU()
                #print(loss.requires_grad)  # 应该为 True
                #print(loss.grad_fn)        # 不应该为 None
                # 计算损失并进行反向传播
                loss.requires_grad_(True)
                loss.backward()
                student_model_optimizer.step()
        print("train_loss:{0} at{1} epoch.".format(loss,epoch))

g_dataset='two'
#gnn_train_dataset = TestbedDataset(root='dataset/pretrain_dataset/', dataset=g_dataset+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
seq_train_dataset=SeqDataset('/media/ext_disk/zhenfang/dataset/pretrain_dataset/processed/'+g_dataset+'_train_sequence.csv')
#smiles_train_dataset=SmileDataset('dataset/pretrain_dataset/processed/'+g_dataset+'_train_smiles.csv')


# In[6]:


import torch_geometric

#gnn_train_dataloader=torch_geometric.data.DataLoader(gnn_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
seq_train_dataloader=torch.utils.data.DataLoader(seq_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
#smiles_train_dataloader=torch.utils.data.DataLoader(smiles_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)

#del seq_train_dataset
# In[18]:
import os
import time
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler

from ModelDistillation import *
import torch
import torch.optim as optim




def main():
    
    protein_embd_dim=320
    #seq_embd_dim=1280
    output_embd_dim=256
    
    
    student_model=SeqMLPStudent(1500,10,320,0.5)
    
    
    student_model.cuda()
    
    
    
    student_model_optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9)                                                                                                       
    #criterion = nn.CrossEntropyLoss().cuda()
    #optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    #scaler = GradScaler()  ###  用于混合精度训练
    dropout=0.5
    for epoch in range(args.epochs):
        trainSeq(epoch,protein_model,seq_train_dataloader,student_model,student_model_optimizer,protein_embd_dim)
    
    
    
    
        if epoch % 50 == 0:
            
            torch.save({
                'model': student_model.state_dict()
                
            }, 'ddp_checkpoint_{}.pt'.format(epoch))
    




# In[20]:
import time
import gc
if __name__ == '__main__':
    
    gc.collect()
    protein_model,protein_alphabet=esm.pretrained.esm2_t6_8M_UR50D()
    
    for param in protein_model.parameters():
            param.requires_grad = False
    
    time_start = time.time()
    main()
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')



# In[ ]:



