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
parser.add_argument('--batch_size', type=int, default=4,
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
            
            #teacher_model_optimizer.zero_grad()
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
        gc.collect()
        loss=0
        inputs_1500_list=[]
        for i, inputs in enumerate(trainloader):
            
                seq_data_list=[]
                seq=inputs
                lenth=len(seq)
                #print('seq type:',type(seq)) 
                
                for m , s in enumerate(seq):
                    seq_data_list.append((str(m),s))
                #student_model_optimizer.zero_grad()
                
                
                del m
                del s
                #inputs.cuda()
                
                student_model_optimizer.zero_grad()
                teacher_outputs = SeqInferTeacher(protein_model,seq_data_list,protein_embd_dim)
                student_outputs = student_model(inputs)
                del seq_data_list
                del inputs
                loss_kd = F.mse_loss(student_outputs.detach(), teacher_outputs.detach())
            
                loss = loss_kd 
            
                print("train_loss:{0} at{1} epoch.".format(loss,epoch))
                getGPU()
                gc.collect()
                #print(loss.requires_grad)  # 应该为 True
                #print(loss.grad_fn)        # 不应该为 None
                # 计算损失并进行反向传播
                loss.requires_grad_(True)
                loss.backward()
                student_model_optimizer.step()
        print("train_loss:{0} at{1} epoch.##############".format(loss,epoch))

g_dataset='two'
#gnn_train_dataset = TestbedDataset(root='dataset/pretrain_dataset/', dataset=g_dataset+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
seq_train_dataset=SeqDataset('/media/ext_disk/zhenfang/dataset/pretrain_dataset/processed/'+g_dataset+'_train_sequence.csv')
#smiles_train_dataset=SmileDataset('dataset/pretrain_dataset/processed/'+g_dataset+'_train_smiles.csv')


# In[6]:


import torch_geometric

#gnn_train_dataloader=torch_geometric.data.DataLoader(gnn_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
#seq_train_dataloader=torch.utils.data.DataLoader(seq_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
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

def prepare():
    

    # 下面几行是新加的，用于启动多进程 DDP
    os.environ['MASTER_ADDR'] = 'localhost'  # 0号机器的IP
    os.environ['MASTER_PORT'] = '19198'  # 0号机器的可用端口
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用哪些GPU
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['WORLD_SIZE'] = str(world_size)
    return args


def init_ddp(local_rank):
    # 有了这一句之后，在转换device的时候直接使用 a=a.cuda()即可，否则要用a=a.cuda(local+rank)
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    #print('local_rank:',local_rank)
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def main(local_rank, args):
    init_ddp(local_rank)  ### 进程初始化
    #model = ConvNet().cuda()  ### 模型的 forward 方法变了
    protein_embd_dim=320
    #seq_embd_dim=1280
    output_embd_dim=256
    ''''
    protein_model,protein_alphabet=esm.pretrained.esm2_t6_8M_UR50D()
    for param in protein_model.parameters():
            param.requires_grad = False
            
    '''  
    protein_model=args.protein_model
    #teacher_model=SeqTeacher(protein_model,320,0.5)
    #teacher_model.cuda()
    student_model=SeqMLPStudent(1500,10,320,0.5)
    
    
    student_model.cuda()
    #student_model = nn.SyncBatchNorm.convert_sync_batchnorm(student_model)  ### 转换模型的 BN 层
    student_model = nn.parallel.DistributedDataParallel(student_model,
                                                device_ids=[local_rank])
    
    #teacher_model = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)  ### 转换模型的 BN 层
    
    #teacher_model = nn.parallel.DistributedDataParallel(teacher_model,
    #                                          device_ids=[local_rank])
    # 创建教师模型的优化器
    #teacher_model_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)

    # 创建学生模型的优化器
    student_model_optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9)                                                                                                       
    #criterion = nn.CrossEntropyLoss().cuda()
    #optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    #scaler = GradScaler()  ###  用于混合精度训练
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        seq_train_dataset)  ### 用于在 DDP 环境下采样
    g = get_ddp_generator()  ###
    train_dloader = torch.utils.data.DataLoader(
        dataset=seq_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  ### shuffle 通过 sampler 完成
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
        generator=g)  ### 添加额外的 generator
    
    dropout=0.5
    for epoch in range(args.epochs):
        if local_rank == 0:  ### 防止每个进程都输出一次
            print(f'begin training of epoch {epoch + 1}/{args.epochs}')
        train_dloader.sampler.set_epoch(epoch)  ### 防止采样出 bug
        trainSeq(epoch,protein_model,train_dloader,student_model,student_model_optimizer,protein_embd_dim)
        if epoch % 50 == 0:
            # 保存模型参数到文件
            #torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(epoch))
            #print(f'begin testing')
            
            torch.save({
                'model': student_model.state_dict(),
                'scaler': scaler.state_dict()
            }, 'ddp_checkpoint_{}.pt'.format(epoch))
    dist.destroy_process_group




# In[20]:
import time
import gc
if __name__ == '__main__':
    # 设置 PYTORCH_CUDA_ALLOC_CONF 环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 示例值，您可以根据需要调整
    torch.cuda.empty_cache()
    gc.collect()
    protein_model,protein_alphabet=esm.pretrained.esm2_t6_8M_UR50D()
    del protein_alphabet
    for param in protein_model.parameters():
            param.requires_grad = False
    args.protein_model=protein_model
    prepare()
    time_start = time.time()
    mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count())
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')



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
parser.add_argument('--batch_size', type=int, default=4,
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
            
            #teacher_model_optimizer.zero_grad()
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
        gc.collect()
        loss=0
        inputs_1500_list=[]
        for i, inputs in enumerate(trainloader):
            
                seq_data_list=[]
                seq=inputs
                lenth=len(seq)
                #print('seq type:',type(seq)) 
                
                for m , s in enumerate(seq):
                    seq_data_list.append((str(m),s))
                #student_model_optimizer.zero_grad()
                
                
                del m
                del s
                #inputs.cuda()
                
                student_model_optimizer.zero_grad()
                teacher_outputs = SeqInferTeacher(protein_model,seq_data_list,protein_embd_dim)
                student_outputs = student_model(inputs)
                del seq_data_list
                del inputs
                loss_kd = F.mse_loss(student_outputs.detach(), teacher_outputs.detach())
            
                loss = loss_kd 
            
                print("train_loss:{0} at{1} epoch.".format(loss,epoch))
                getGPU()
                gc.collect()
                #print(loss.requires_grad)  # 应该为 True
                #print(loss.grad_fn)        # 不应该为 None
                # 计算损失并进行反向传播
                loss.requires_grad_(True)
                loss.backward()
                student_model_optimizer.step()
        print("train_loss:{0} at{1} epoch.##############".format(loss,epoch))
def trainSeqStudent(epoch,trainloader,student_model,student_model_optimizer,protein_embd_dim):
        gc.collect()
        loss=0
        inputs_1500_list=[]
        for i, inputs in enumerate(trainloader):
            
                
                student_model_optimizer.zero_grad()
                
                student_outputs = student_model(inputs)
                
                
            
                print("train_output:{0} at {1} epoch.".format(student_outputs.shape,epoch))
                getGPU()
                gc.collect()
                
        print("train_loss:{0} at {1} epoch.##############".format(student_outputs.shape,epoch))
g_dataset='two'
#gnn_train_dataset = TestbedDataset(root='dataset/pretrain_dataset/', dataset=g_dataset+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
seq_train_dataset=SeqDataset('/media/ext_disk/zhenfang/dataset/pretrain_dataset/processed/'+g_dataset+'_train_sequence.csv')
#smiles_train_dataset=SmileDataset('dataset/pretrain_dataset/processed/'+g_dataset+'_train_smiles.csv')


# In[6]:


import torch_geometric

#gnn_train_dataloader=torch_geometric.data.DataLoader(gnn_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
#seq_train_dataloader=torch.utils.data.DataLoader(seq_train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
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

def prepare():
    

    # 下面几行是新加的，用于启动多进程 DDP
    os.environ['MASTER_ADDR'] = 'localhost'  # 0号机器的IP
    os.environ['MASTER_PORT'] = '19198'  # 0号机器的可用端口
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用哪些GPU
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['WORLD_SIZE'] = str(world_size)
    return args


def init_ddp(local_rank):
    # 有了这一句之后，在转换device的时候直接使用 a=a.cuda()即可，否则要用a=a.cuda(local+rank)
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    #print('local_rank:',local_rank)
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def main(local_rank, args):
    init_ddp(local_rank)  ### 进程初始化
    #model = ConvNet().cuda()  ### 模型的 forward 方法变了
    protein_embd_dim=320
    #seq_embd_dim=1280
    output_embd_dim=256
    
   
            
    
    student_model=SeqMLPStudent(1500,10,320,0.5)
    
    
    student_model.cuda()
    #student_model = nn.SyncBatchNorm.convert_sync_batchnorm(student_model)  ### 转换模型的 BN 层
    student_model = nn.parallel.DistributedDataParallel(student_model,
                                                device_ids=[local_rank])
    
    
    student_model_optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9)                                                                                                       
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        seq_train_dataset)  ### 用于在 DDP 环境下采样
    g = get_ddp_generator()  ###
    train_dloader = torch.utils.data.DataLoader(
        dataset=seq_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  ### shuffle 通过 sampler 完成
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
        generator=g)  ### 添加额外的 generator
    
    dropout=0.5
    for epoch in range(args.epochs):
        if local_rank == 0:  ### 防止每个进程都输出一次
            print(f'begin training of epoch {epoch + 1}/{args.epochs}')
        train_dloader.sampler.set_epoch(epoch)  ### 防止采样出 bug
        trainSeqStudent(epoch,train_dloader,student_model,student_model_optimizer,protein_embd_dim)
        if epoch % 50 == 0:
            # 保存模型参数到文件
            #torch.save(model.state_dict(), 'model_epoch_{}.pth'.format(epoch))
            #print(f'begin testing')
            
            pirnt('epoch50##########:',epoch)
    dist.destroy_process_group




# In[20]:
import time
import gc
if __name__ == '__main__':
    # 设置 PYTORCH_CUDA_ALLOC_CONF 环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 示例值，您可以根据需要调整
    torch.cuda.empty_cache()
    gc.collect()
    
    prepare()
    time_start = time.time()
    mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count())
    time_elapsed = time.time() - time_start
    print(f'\ntime elapsed: {time_elapsed:.2f} seconds')



# In[ ]:




