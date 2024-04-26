from loader import MoleculeDataset, SeqDataset,SeqMolDataset,SmileDataset#########################
import torch
import torch
#import args
from torchvision.models import resnet18

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

from loader import MoleculeDataset#################
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred,GNN_graphpred_1
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split,scaffold_split_1
import pandas as pd
import os
import shutil
from tensorboardX import SummaryWriter
import esm

from SeqMolModel import InteractionModel,InteractionModel_1,SequenceModel,InteractionModel_4


# Training settings
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')#0000
parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
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
parser.add_argument('--dataset', type=str, default = 'affinity', help='root directory of dataset. For now, only classification.')
#parser.add_argument('--input_model_file', type=str, default = 'None', help='filename to read the model (if there is any)')
parser.add_argument('--input_model_file', type=str, default = 'Mole-BERT', help='filename to read the model (if there is any)')
parser.add_argument('--filename', type=str, default = '', help='output filename')
parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')

parser.add_argument('--n_head', type=int, default = 12, help='number of workers for dataset loading')

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


#load pretained model of Mole-Bert


device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.runseed)

num_tasks=1
# Load ESM-2 model
protein_model, protein_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
protein_model.to(device)
#freezing parameters
for i,p in enumerate(protein_model.parameters()):
    p.requires_grad = False
#print(protein_alphabet)
#alphabet = esm.Alphabet.from_architecture(model_data["args"].arch)
#batch_converter = alphabet.get_batch_converter()
protein_model.eval()  # disables dropout for deterministic results

#self.molecular.model,self.molecular.node_representation,self.molecular.features = GNN_graphpred_1(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
molecular_model = GNN_graphpred_1(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
###################################
if not args.input_model_file == "None":###############
    print('Not from scratch')
    molecular_model.from_pretrained('model_gin/{}.pth'.format(args.input_model_file))
    print('rese:model_gin')
molecular_model.to(device)
for i,p in enumerate(molecular_model.parameters()):
    p.requires_grad = False#freezing parameters
#freezing parameters
for i,p in enumerate(protein_model.parameters()):
    p.requires_grad = False#freezing parameters



model_param_group = []
model_param_group.append({"params": molecular_model.gnn.parameters()})
if args.graph_pooling == "attention":
    model_param_group.append({"params": molecular_model.pool.parameters(), "lr":args.lr*args.lr_scale})
'''
————————————————
版权声明：本文为CSDN博主「Dreamcatcher风」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Wind_2028/article/details/120541017   
'''



model_param_group.append({"params": molecular_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
optimizer = optim.Adam(model_param_group, lr=0.01, weight_decay=args.decay)#############




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
#from utils import normalize_smiles
# create a function (this my favorite choice)
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
    
    
margs = args
tokenizer = MolTranBertTokenizer('finetune/bert_vocab.txt')
seed.seed_everything(margs.seed)
if margs.seed_path == '':
    #print("# training from scratch")
    smile_model = LightningModule(margs, tokenizer)
else:
    #print("# loaded pre-trained model from {args.seed_path}")
    smile_model = LightningModule(margs, tokenizer).load_from_checkpoint(margs.seed_path, strict=False, config=margs, tokenizer=tokenizer, vocab=len(tokenizer.vocab))#########################33

#print('model:',smile_model)
#freezing parameters
for i,p in enumerate(smile_model.parameters()):
    p.requires_grad = False
    
#num_tasks=1
model= InteractionModel_4(protein_model=protein_model,molecular_model=molecular_model,smile_model=smile_model,protein_embd_dim=1280,num_tasks=1,device=device,mol_embd_dim=300,smile_embd_dim=768) 
model.to(device)
#print(model)#nice


###################when doing geometric.data process, if there is some changes, please delete the directory process and let it generate again

gnn_dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)###########################转换成了分子图的格式

seq_dataset=SeqDataset('dataset/affinity/processed/sequence.csv')
#smiles_dataset=SmileDataset('dataset/affinity/processed/smiles.csv')

def collate(batch):
       
        tokenizer = MolTranBertTokenizer('finetune/bert_vocab.txt')
        
        tokens = tokenizer.batch_encode_plus([ smile for smile in batch], padding=True, add_special_tokens=True)
       
        return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']))
    
    
smiles_dataset=SmileDataset('dataset/affinity/processed/smiles.csv')

        
    
#####test with chartGPT  DataSet extends two parents' classes

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

        
###########split train dataset validate dataset test dataset
seq_gnn_dataset=MultiDatasetMixin(seq_dataset,gnn_dataset,smiles_dataset)

#print('seq_dataset:',seq_dataset)

#seq_dataloader=DataLoader(seq_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
'''
for i ,seq in enumerate(seq_dataloader):
    print(seq)
'''
if args.split == "scaffold":
        smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        
        #print('smiles_list:',smiles_list)
        mol_train_dataset, mol_valid_dataset, mol_test_dataset ,seq_train_dataset,seq_valid_dataset,seq_test_dataset,smile_train_dataset,smile_valid_dataset,smile_test_dataset= scaffold_split_1(seq_gnn_dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)##########dataset
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

#print('++++++++++', mol_train_dataset[0])


#seq_mol_train_dataset=MultiDatasetMixini(seq_train_dataset,mol_train_dataset)
#seq_mol_valid_dataset=SeqMolDataset(seq_valid_dataset,mol_valid_dataset)
#seq_mol_test_dataset=SeqMolDataset(seq_train_dataset,mol_test_dataset)
seq_train_dataloader1 = DataLoader(seq_train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)######False
mol_train_dataloader2 = GeometricDataLoader(mol_train_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
smile_train_dataloader3=DataLoader(smile_train_dataset,batch_size=args.batch_size,collate_fn=collate, shuffle=False,num_workers=args.num_workers)
seq_mol_smile_train_multi_loader = MultiDataLoader(seq_train_dataloader1, mol_train_dataloader2,smile_train_dataloader3)
# Set the shuffle parameter simultaneously for both dataloaders
seq_mol_smile_train_multi_loader.set_shuffle(True)
'''
print('seq_train_dataset#########:',seq_train_dataset)

for i,seq in enumerate(seq_train_dataloader1):
    print(seq)
'''
seq_valid_dataloader1 = DataLoader(seq_valid_dataset, batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)######False
mol_valid_dataloader2 = GeometricDataLoader(mol_valid_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
smile_valid_dataloader3=DataLoader(smile_valid_dataset, batch_size=args.batch_size,collate_fn=collate,shuffle=False,num_workers=args.num_workers)
for i,m in enumerate(smile_train_dataloader3):
    #print('m@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:',m)
    break
seq_mol_smile_valid_multi_loader = MultiDataLoader(seq_valid_dataloader1, mol_valid_dataloader2,smile_valid_dataloader3)
# Set the shuffle parameter simultaneously for both dataloaders
seq_mol_smile_valid_multi_loader.set_shuffle(True)

seq_test_dataloader1 = DataLoader(seq_test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)######False
mol_test_dataloader2 = GeometricDataLoader(mol_test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
smile_test_dataloader3=DataLoader(smile_test_dataset,batch_size=args.batch_size,collate_fn=collate, shuffle=False,num_workers=args.num_workers)

seq_mol_smile_test_multi_loader = MultiDataLoader(seq_test_dataloader1, mol_valid_dataloader2,smile_test_dataloader3)
# Set the shuffle parameter simultaneously for both dataloaders
seq_mol_smile_test_multi_loader.set_shuffle(True)


def train(args, epoch, model, device, loader, optimizer):
    model.train()
    #epoch_iter = tqdm(loader, desc="Iteration")
    for step, (A,B,C) in enumerate(loader):
        
        seq_data_list=[]
        seq=A
        lenth=len(seq)
        for m , s in enumerate(seq):
            seq_data_list.append((str(m),s))
        B=B.to(device)
        D,E=C
        D=D.to(device)
        E=E.to(device)
        C=(D,E)
        #print('D!!!!!!!!!!!!!!!!:',D)
        #print('E#####################:',E)
        
        pred=model(seq_data_list,B,C)#model is error
        y_true = B.y.view(pred.shape).to(torch.float64)
        loss = criterion(pred, y_true)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        #epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f}")
        if (epoch + 1) % 10 == 0:
            print(f'training Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
            
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
        
        
            pred=model(seq_data_list,B,C)#model is error
            y_true=B.y.view(pred.shape).to(torch.float64)
            val_loss = criterion(pred, y_true)
            print(f'Validation Loss: {val_loss.item():.4f}')    
                             

    
    
    return val_loss.item()


import torch, gc
#criterion = nn.BCEWithLogitsLoss(reduction = "none")
criterion=nn.SmoothL1Loss()
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
        
    train(args, epoch, model, device, seq_mol_smile_train_multi_loader, optimizer)

    print("====Evaluation")
    if args.eval_train:
        train_loss = eval(args, model, device, seq_mol_smile_train_multi_loader)
    else:
        print("omit the training accuracy computation")
        train_loss = 0
    val_loss = eval(args, model, device, seq_mol_smile_valid_multi_loader)
    test_loss = eval(args, model, device, seq_mol_smile_test_multi_loader)

    print("train: %f val: %f test: %f" %(train_loss, val_loss, test_loss))
    