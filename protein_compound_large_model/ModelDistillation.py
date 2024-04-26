#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
from model import GNN, GNN_graphpred,GNN_graphpred_1
from fast_transformers.masking import LengthMask as LM
import torch
import torch
from torchvision.models import resnet18

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

from loader import MoleculeDataset#################
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred,GNN_graphpred_1
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd
import os
import shutil
from tensorboardX import SummaryWriter
import esm
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
class SeqTeacher(nn.Module):
    def __init__(self,protein_model,protein_embd_dim,output_embd_dim,dropout):
        super(SeqTeacher,self).__init__()
        self.protein_embd_dim=protein_embd_dim
        self.output_embd_dim=output_embd_dim
        #self.protein_model, self.protein_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        #self.protein_model,self.protein_alphabet=esm.pretrained.esm2_t12_35M_UR50D()
        #self.alphabet=self.protein_model.alphabet##########.to(device)
        self.protein_model=protein_model
        self.alphabet=self.protein_model.alphabet
        self.batch_converter=self.alphabet.get_batch_converter()
        #self.protein_sequence_representations = torch.tensor([1,2])
        #self.protein_linear=torch.nn.Linear(protein_embd_dim,output_embd_dim)
        #self.relu=nn.ReLU()
        #self.dropout = nn.Dropout(dropout)
        #self.device=device
        self.results=None
        '''
        print('frozening parameters')
        for param in self.protein_model.parameters():
            param.requires_grad = False
        '''
    def forward(self,x):
        
        print('starting$$$$$$$$$$$$:')
        #print('x:',x)
        #self.protein_model.cuda()
        #self.alphabet.cuda()
        self.protein_model.cuda()
        batch_labels, batch_strs, batch_tokens = self.batch_converter(x)#############
        
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        #print('batch_strs:',batch_strs)
        #batch_labels.to(self.device)
        batch_tokens=batch_tokens.cuda()######################
        print('batch_tokens:',batch_tokens.device.type)
        batch_lens.cuda()
        print('protein_batch_lens:',len(batch_lens),batch_lens)
        #print('batch_tokens:',batch_tokens)
        self.protein_model.cuda()
        '''
        with torch.no_grad():
            #batch_tokens.to(self.device)
            #batch_lens.to(self.device)
            #results = self.protein_model(batch_tokens, repr_layers=[33], return_contacts=True)###############3
            print('into torch.no_grad()')
            #self.results = self.protein_model(batch_tokens, repr_layers=[12], return_contacts=True)###############3
            self.results = self.protein_model(batch_tokens, repr_layers=[6], return_contacts=True)###############3
        '''
        self.results = self.protein_model(batch_tokens, repr_layers=[6], return_contacts=True)
        protein_token_representations = self.results["representations"][6]
        m,n,v=protein_token_representations.shape
        print('m,n,v:',m,n,v)
        print('protein_token_representations:',protein_token_representations.shape)
        #protein_token_representations.to(device)
        #protein_token_representations.to(self.device)
        for i, tokens_len in enumerate(batch_lens):
            if i==0:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,self.protein_embd_dim)
                output=u
                
            else:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,self.protein_embd_dim)
                output=torch.cat([output,u],dim=0)
        print('output_1:',output.shape)
        output=self.dropout(self.relu(self.protein_linear( output)))
        print('output:',output.shape)
        return output

def SeqInferTeacher(protein_model,x,protein_embd_dim):
    
        
        
        #print('starting$$$$$$$$$$$$:')
        
        for param in protein_model.parameters():
            param.requires_grad = False
        alphabet=protein_model.alphabet
        batch_converter=alphabet.get_batch_converter()
        protein_model.cuda()
        batch_labels, batch_strs, batch_tokens = batch_converter(x)#############
        del batch_labels
        del batch_strs
        del x
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        del alphabet
        gc.collect()
        torch.cuda.empty_cache()
        batch_tokens=batch_tokens.cuda()######################
        #print('batch_tokens:',batch_tokens.device.type)
        batch_lens.cuda()
        #print('protein_batch_lens:',len(batch_lens),batch_lens)
        #print('batch_tokens:',batch_tokens)
        #self.protein_model.cuda()
        results=None
        #print('inference=========')       
        with torch.no_grad():
            #results = protein_model(batch_tokens, repr_layers=[6], return_contacts=True)
            results = protein_model(batch_tokens, repr_layers=[6], return_contacts=False)
        #print('results===============')
        
        protein_token_representations = results["representations"][6]
        del protein_model
        protein_token_representations=protein_token_representations.to(torch.float16)
        del results
        del batch_tokens
        
        #gc.collect()
        #torch.cuda.empty_cache()
        #del protein_model
        gc.collect()
        torch.cuda.empty_cache()
        m,n,v=protein_token_representations.shape
        #print('m,n,v:',m,n,v)
        #print('protein_token_representations:',protein_token_representations.shape)
        #protein_token_representations.to(device)
        #protein_token_representations.to(self.device)
        output=None
        u=None
        for i, tokens_len in enumerate(batch_lens):
            if i==0:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,protein_embd_dim)
                output=u
                
            else:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,protein_embd_dim)
                output=torch.cat([output,u],dim=0)
        print('output_1:',output.shape)
        #print('output_protein.device:',output.device)
        del protein_token_representations
        del u
        print('ModelDistillation#############')
        gc.collect()
        torch.cuda.empty_cache()
        #getGPU()
        
        return output

class MolecularTeacher(nn.Module):
    def __init__(self,checkpoint_file,num_layer,mole_embd_dim,output_embed_dim,num_task,JK,dropout,graph_poolingg,gnn_type):
        super(MolecularTeacher,self).__init__()
        
        self.molecular_model = GNN_graphpred_1(num_layer,emb_dim, num_tasks, JK, dropout, graph_pooling, gnn_type)
        self.molecular_model.from_pretrained('model_gin/{}.pth'.format(checkpoint_file))
            
        
        
        
        self.mole_model=self.molecular_model.gnn#nice微调，必须写成类成员  self，否则就不能微调
        self.mole_pool=self.molecular_model.pool#nice微调，必须写成类成员  self，否则就不能微调
        self.mole_linear=torch.nn.Linear(mol_embd_dim,output_embd_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self,molecular_data):
        #model=self.molecular_model.from_pretrained('model_gin/{}.pth'.format(args.input_model_file))
        y=self.mole_model.gnn(molecular_data.x,molecular_data.edge_index,molecular_data.edge_attr)
        y=self.mole_pool(y,molecular_data.batch)
        y=self.relu(self.dropout(self.mole_linear(y)))
        
        return y
    
    
    
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
#from finetune.tokenizer.tokenizer import MolTranBertTokenizer
from fast_transformers.masking import LengthMask as LM
#from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
from apex import optimizers
sys.path.append('finetune/')
from utilss import normalize_smiles
from tokenizer.tokenizer import MolTranBertTokenizer
from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
class LightningModule(pl.LightningModule):

    def __init__(self, config, tokenizer):
        super(LightningModule, self).__init__()

        self.config = config
        self.save_hyperparameters(config)
        self.tokenizer=tokenizer
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

from fast_transformers.masking import LengthMask as LM    
class SmileTeacher(nn.Module):
    def __init__(self,smile_embd_dim,output_embd_dim,margs,seed_path,config,tokenizer,vocab,strict):
        super(SmileTeacher,self).__init__()
        self.margs = margs
        self.tokenizer = MolTranBertTokenizer('finetune/bert_vocab.txt')
        self.smile_model=None
        if margs.seed_path == '':
            
            self.smile_model = LightningModule(self.margs, self.tokenizer)
        else:
            
            self.smile_model = LightningModule(margs, tokenizer).load_from_checkpoint(seed_path,  config, tokenizer, vocab,strict)#########################33
        self.smile_linear=nn.Linear(smile_embd_dim,output_embd_dim)
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, smile_data):
        x,mask=smile_data
        length_mask=LM(mask.sum(-1))
        
        with torch.no_grad():
            token_embeddings = self.smile_model.tok_emb(x) # each index maps to a (learnable) vector
            x6 = self.smile_model.drop(token_embeddings)
            x7 = self.smile_model.blocks(x6, length_mask=LM(mask.sum(-1)))
            token_embeddings = x7
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            x4 = sum_embeddings / sum_mask####################################
        x4=self.relu(self.dropout(self.smile_linear(x4)))
        return x4
        
        

class SeqMLPStudent(nn.Module):
    
    def __init__(self,input_seq_embd_dim,embding_size,output_seq_embd_dim,dropout):
        super(SeqMLPStudent, self).__init__()
        self.amino_acid_to_index = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q':6,'E':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20}
        self.embedding = nn.Embedding(21, embding_size)
        '''
        self.fc = nn.Sequential(
            nn.Linear(embding_size, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, output_seq_embd_dim)
        )
        '''
        #self.device=device
        self.dropout=nn.Dropout()
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm1d(input_seq_embd_dim)
        self.bn2=nn.BatchNorm1d(512)
        #self.linear1=nn.Linear(input_seq_embd_dim*embding_size,512)
        self.linear1=nn.Linear(input_seq_embd_dim,512)
        self.linear2=nn.Linear(512,320)
        #self.linear3=nn.Linear(256,out_seq_embd_dim)

        

    def forward(self, x):
        # 将字符串输入编码为词嵌入
        #print('x:',len(x))
        #print('x0:',x[1])
        indice=[]
        
        indic_1500=[0]*1500
        for i,s in enumerate(x):
                indic = [self.amino_acid_to_index[aa] for aa in s]
                lenth=len(indic)
                if len(indic)<1500:
                    indic_1500[:lenth]=indic
                indice.append(indic_1500)
                
                
        indices=np.array(indice,dtype=np.int)
                
        m = torch.tensor(indices, dtype=torch.float32)
        m=m.cuda()
        embedded=m
        #m.to(self.device)
        #print('m.device:',m.shape)
        #embedded = self.embedding(m)#######################################
        #embedded=embedded.to(torch.float16)
        #print('embedding.device:',embedded.shape)
        # 将词嵌入展平为一维张量
        #embedded = embedded.view(embedded.size(0), -1)#########################
        #print('embedded.size:',embedded.shape)
        #y = embedded.reshape(embedded.size(0), -1)
        output=self.dropout(self.relu(self.linear1(self.bn1(embedded))))
        output=self.dropout(self.relu(self.linear2(self.bn2(output))))
        print('SeqTeacherModel#######################')
        getGPU()
        
        return output

    
class MolMLPStudent(nn.Module):
    
    def __init__(self,input_mol_embd_dim,output_mol_embd_dim):
        super(MolMLPDistillation, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_mol_embd_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_mol_embd_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class SmileMLPStudent(nn.Module):
    
    def __init__(self,input_smile_embd_dim,output_smile_embd_dim):
        super(SmileMLPDistillation, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_smile_embd_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_smile_embd_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim1, input_dim2):
        super(CrossAttentionLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim1, input_dim2)###################妙
        self.linear2 = nn.Linear(input_dim2, input_dim1)

    def forward(self, x1, x2,d1,d2):#d1=d2=d=output_embd_dim
        # 计算注意力权重
        #print('x1,x2:',x1.shape,x2.shape)
        attn_weights2 = torch.matmul(self.linear1(x1), x2.transpose(0, 1))
        #print('attn_wights:',attn_weights.shape)
        #attn_weights = attn_weights.squeeze(dim=2)
        attn_weights2 = nn.functional.softmax(attn_weights2, dim=1)
        
        # 使用注意力权重加权融合两个向量
        fused_x2 = torch.matmul(attn_weights2, x2)/np.sqrt(d2)
        #fused_x1 = torch.matmul(attn_weights2, x2)/np.sqrt(d2)
        #print('fused_x1:',fused_x1.shape)
        
        attn_weights1 = torch.matmul(self.linear2(x2), x1.transpose(0, 1))
        #print('attn_wights:',attn_weights.shape)
        #attn_weights = attn_weights.squeeze(dim=2)
        attn_weights1 = nn.functional.softmax(attn_weights1, dim=1)
        fused_x1 = torch.matmul(attn_weights1, x1)/np.sqrt(d1)
        #fused_x2 = torch.matmul(attn_weights1.transpose(0, 1), x1)/np.sqrt(d1)
        #print('fused_x2:',fused_x2.shape)
        return fused_x1, fused_x2

    
    
class InteractionModel_4(torch.nn.Module):
    def __init__(self, protein_model,molecular_model,smile_model,protein_embd_dim,mol_embd_dim,num_tasks,output_embd_dim, device,smile_embd_dim, dropout=0.2,aggr = "mean"):
        super( InteractionModel_4, self ).__init__() 
        self.protein_embd_dim=protein_embd_dim
        self.mol_embd_dim=mol_embd_dim
        self.smile_embd_dim=smile_embd_dim
        self.output_embd_dim=output_embd_dim
        
        self.protein_model=protein_model.to(device)
        self.mole_model=molecular_model
        self.smile_model=smile_model
        
        self.pred_linear = torch.nn.Linear(2*output_embed_dim, num_tasks)
        self.relu=nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        
        self.mol_cross_attention = CrossAttentionLayer(protein_embd_dim, mol_embd_dim)
        self.smile_cross_attention = CrossAttentionLayer(protein_embd_dim, smile_embd_dim)
        
        
        self.layer_norm = nn.LayerNorm(output_embd_dim, eps=1e-6)  # 默认对最后一个维度初始化
       
        
        
        
        
        self.relu1 = nn.GELU()
        
        self.dropout = nn.Dropout(p=0.3)  # dropout训练
        # 定义可学习参数 t
        
        
        
        self.sigmoid=nn.Sigmoid()
        
        self.t = nn.Parameter(torch.Tensor(1))
        self.t_linear1=nn.Linear(1,50)
        self.t_linear2=nn.Linear(50,1)
        self.t.data.fill_(0.5)  # 初始化 t 为0.5
        
    #def forward(self, protein_data,*molecular_data):#******不能有，否则出错
    def forward(self, protein_data,molecular_data,smile_data):#
        x1=self.protein_model(protein_data)
        x2=self.mole_model(molecular_data)
        x3=self.protein_model(protein_data)
        x4=self.smile_model(smile_data)
        
        
        
        x1=self.layer_norm(x1)
        x2=self.layer_norm(x2)
        x3=self.layer_norm(x3)
        x4=self.layer_norm(x4)
        fused_x1, fused_x2 =  self.mol_cross_attention(x1, x2,self.output_embd_dim,self.output_embd_dim)  
        
        
        x11_att=fused_x1
        x22_att=fused_x2
        
        x11=torch.add(x11_att,x1)
        x22=torch.add(x22_att,x2)
        # 合并两个融合后的向量
        output12 = torch.cat([x11, x22], dim=1)   
        out12 = self.pred_linear(self.dropout(self.relu(out12)))
        
        
        
        
        
        fused_x3, fused_x4 =  self.smile_cross_attention(x3, x4,self.protein_embd_dim,self.smile_embd_dim)  
        x33_att=fused_x3
        x44_att=fused_x4
        x33=torch.add(x33_att,x3)
        x44=torch.add(x44_att,x4)
        # 合并两个融合后的向量
        output34 = torch.cat([x33, x44], dim=1)   
        out34 = self.pred_linear(self.dropout(self.relu(out34)))
        
        
        
         # 融合两个分支特征
        #t1=self.relu(self.t_linear1(self.t))
        t2=torch.sigmoid(self.t_linear2(t1))
        
        out = t2 * out12 + (1 - t2) * out34
        
        
        
        return out


# In[ ]:




