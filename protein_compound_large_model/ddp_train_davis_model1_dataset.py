#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.cuda.amp import GradScaler
from sklearn.metrics import roc_auc_score
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from loader1 import myMoleculeDataset,MoleculeDataset,MoleculeDatasetBig, SeqDataset,SeqMolDataset,SmileDataset#########################
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
print(torch.cuda.is_available())
import torch
torch.cuda.current_device()
torch.cuda._initialized = True
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# Training settings
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int,default=0,
                        help='which gpu to use if any (default: 0)')#0000
parser.add_argument('--gpu',default='0,1,2,3,4,5,6')
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


def init_ddp(local_rank):
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '19198'  
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    #print('init_ddp_local_rank:',local_rank)
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    dist.init_process_group(backend='nccl', init_method='env://')
    
def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    print('local_rank_get_ggp_generator:',local_rank)
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
    #print('collate_batch:',batch)
    #print('collate_batch_0:',batch[0])
    tokenizer = MolTranBertTokenizer('finetune/bert_vocab.txt')
        
    tokens = tokenizer.batch_encode_plus([ smile for smile in batch], padding=True, add_special_tokens=True)
    #print('tokens[1]_mask:',tokens[1])
    #print('colate###########################')
    #for i,m in enumerate(tokens):
            
    #print('collate_tokens_input_ids:',tokens['input_ids'])
    #print('colate_tokens_attention_mask:',tokens['attention_mask'])
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
    def set_epoch(self, epoch):
        self.dataloader1.sampler.set_epoch(epoch) 
        self.dataloader2.sampler.set_epoch(epoch)
        self.dataloader3.sampler.set_epoch(epoch)

        


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
        #print(C)
        D,E=C
        D=D.to(device)
        E=E.to(device)
        C=(D,E)
        
        pred=model(seq_data_list,B,C)#model is error
        
        y_true = B.y.view(pred.shape).to(torch.float32)
        
        loss1=criterion(pred,y_true)
        
        loss=loss1
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        scaler.scale(loss).backward()  ###
        scaler.step(optimizer)  ###
        scaler.update()
        print("train: %f " %(loss))
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
            val_loss = criterion(pred, y_true)
            #print(f'Validation Loss: {val_loss.item():.4f}')    
                             

   
    
    return val_loss.item()


import torch, gc

def main(local_rank,args):
    
    
    # 初始化分布式训练环境
    #init_distributed()
    #init_distributed_mode(args)
    print('distributed#########')
    # 创建模型
    init_ddp(local_rank) ############
    #print('init_ddp_rank:',local_rank)
    
    gnn_dataset = myMoleculeDataset(root="./dataset/" + args.dataset)###########################转换成了分子图的格式
    print('args.dataset:',args.dataset)
    seq_dataset=SeqDataset('dataset/davis/processed/sequence.csv')
    smiles_dataset=SmileDataset('dataset/davis/processed/smiles.csv')
    
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
    print('model%%%%%%%%%%%%%%%%')
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    print('nn.SyncBatchNorm.convert_sync_batchnorm(model)################')
    # 将模型封装为DistributedDataParallel对象
    model = DistributedDataParallel(model,device_ids=[local_rank]) 
    print('DistributedDataParallled(model)@@@@@@@@@@@@')                                                        
    seq_gnn_smile_dataset=MultiDatasetMixin(seq_dataset,gnn_dataset,smiles_dataset)
    # 加载数据集
    if args.split == "scaffold":
        smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        
        #print('smiles_list:',smiles_list)
        gnn_train_dataset, gnn_valid_dataset, gnn_test_dataset ,seq_train_dataset,seq_valid_dataset,seq_test_dataset,smiles_train_dataset,smiles_valid_dataset,smiles_test_dataset= scaffold_split_1(seq_gnn_smile_dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)##########dataset
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        #print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        #print("random scaffold")
    else:
        raise ValueError("Invalid split option.")
    
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
    
    
    smile_train_dataloader3=torch.utils.data.DataLoader(smiles_train_dataset,                                                                generator=g,sampler=smiles_train_sampler,pin_memory=True,batch_size=args.batch_size,collate_fn=collate_fn, shuffle=False,num_workers=args.num_workers) 
                                                                    
    seq_mol_smile_train_multi_loader = MultiDataLoader(seq_train_dataloader1, mol_train_dataloader2,smile_train_dataloader3)
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
    
    criterion=nn.SmoothL1Loss()
    mse_criterion=nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()  ###
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
        #train(model, seq_mol_smile_train_multi_loader, criterion, optimizer, scaler)
        train(args, epoch, model,  seq_mol_smile_train_multi_loader, optimizer,scaler)
    if local_rank == 0:
        print('begin testing')
    test(model, seq_mol_smile_test_multi_loader)
    
    if local_rank == 0:  ##
        print("train: %f  test: %f" %(train_loss, test_loss))
        if epoch %50==0:
            torch.save({
                'model': model.state_dict(),
                'scaler': scaler.state_dict()
            }, 'results/davis/model1/ddp_davis_model1_checkpoint_{}.pt'.format(epoch))
    dist.destroy_process_group()
    test_loss = eval(args, model, device, seq_mol_smile_test_multi_loader)
    print("test: %f" %(test_loss))
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
    #args = prepare()
    print('cuda count:',torch.cuda.device_count())
    time_start = time.time()
    mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count())
    time_elapsed = time.time() - time_start
    print('\ntime elapsed: {time_elapsed:.2f} seconds')
        

