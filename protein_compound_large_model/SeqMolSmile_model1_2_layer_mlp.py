#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

from fast_transformers.masking import LengthMask as LM
import torch
import torch
from torchvision.models import resnet18

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import gc
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

from splitters import scaffold_split
import pandas as pd
import os
import shutil
from tensorboardX import SummaryWriter
import esm

# In[3]:


class InteractionModel(torch.nn.Module):
    def __init__(self, protein_model,molecular_model,mol_emb_dim,pro_emb_dim,num_tasks, aggr = "add"):
        super( InteractionModel, self ).__init__() 
        
        self.batch_converter=protein_model.alphabet.get_batch_converter()
        
        #self.molecular.node_representation=molecular.model.gnn()
        #self.molecular.model,self.molecular.node_representation,self.molecular.features = GNN_graphpred_1(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        self.protein_sequence_representations = []
        self.molecular_model=molecular_model############微调，必须写成类成员  self，否则就不能微调
        self.pred_linear = torch.nn.Linear(pro_emb_dim+mol_emb_dim, num_tasks)
        
        def forward(self, protein_data,*molecular_data):
            ############protein
            batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = protein_model(batch_tokens, repr_layers=[33], return_contacts=True)
        
            protein_token_representations = results["representations"][33]
            
            for i, tokens_len in enumerate(batch_lens):
                self.protein_sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            
            
            ################molecular
            
            #set up optimizer
            #different learning rate for different part of GNN
            molecular_node_representation=self.molecular_model.gnn(molecular_data[0],molecular_data[1],molecular_data[2])
            molecular_representation=self.molecular_model.pool(molecular.node_representation,molecular_data[3])
            #molecular_representation=self.molecular.features(*molecular_data)
            
            
            
            all_features=torch.concat([self.protein_sequence_representations,molecular_representation],axis=1)
            self.pred_linear(all_features)

class InteractionModel_2(torch.nn.Module):
    def __init__(self, protein_model,molecular_model,mol_emb_dim,pro_emb_dim,num_tasks, device,aggr = "add"):
        super( InteractionModel_2, self ).__init__() 
        self.protein_model=protein_model.to(device)
        self.alphabet=protein_model.alphabet##########.to(device)
        self.batch_converter=self.alphabet.get_batch_converter()
        self.device=device
        #self.molecular.node_representation=molecular.model.gnn()
        #self.molecular.model,self.molecular.node_representation,self.molecular.features = GNN_graphpred_1(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        self.protein_sequence_representations = []
        #self.molecular_model############微调，必须写成类成员  self，否则就不能微调
        self.mole_model=molecular_model.gnn#nice微调，必须写成类成员  self，否则就不能微调
        self.mole_pool=molecular_model.pool#nice微调，必须写成类成员  self，否则就不能微调
        self.pred_linear = torch.nn.Linear(pro_emb_dim+mol_emb_dim, num_tasks)
        
    #def forward(self, protein_data,*molecular_data):#******不能有，否则出错
    def forward(self, protein_data,molecular_data):#
        ############protein
        print('self.device:',self.device)
        print('protein_data:',protein_data)
            
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_data)
        batch_tokens=batch_tokens.to(self.device)######################
        print('batch_tokens:',batch_tokens.device.type)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        print('protein_batch_lens:',len(batch_lens))
        with torch.no_grad():
            results = self.protein_model(batch_tokens, repr_layers=[33], return_contacts=True)###############3
        
        protein_token_representations = results["representations"][33]
        print('protein_token_representations:',protein_token_representations.shape)
        for i, tokens_len in enumerate(batch_lens):
            self.protein_sequence_representations.append(protein_token_representations[i, 1 : tokens_len - 1].mean(0))
            
        print('self.protein_sequence_representations:',self.protein_sequence_representations)
        ################molecular
            
        #set up optimizer
        #different learning rate for different part of GNN
        molecular_node_representation=self.mole_model(molecular_data.x,molecular_data.edge_index,molecular_data.edge_attr)
        print('molecular_node_representation:',molecular_node_representation)
        molecular_representation=self.mole_pool(molecular_node_representation,molecular_data.batch)
        print('molecular_representation:',molecular_representation.shape)
        print('molecular_batch:',len(molecular_data.batch))
        #molecular_representation=self.molecular.features(*molecular_data)
            
            
            
        all_features=torch.concat([self.protein_sequence_representations,molecular_representation],axis=1)
        print('all_features:',all_features)
        out=self.pred_linear(all_features)
        return out


'''    
#CrossAttention
class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim1, input_dim2):
        super(CrossAttentionLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim1, input_dim2)###################妙
        self.linear2 = nn.Linear(input_dim2, input_dim1)

    def forward(self, x1, x2,d):
        # 计算注意力权重
        #print('x1,x2:',x1.shape,x2.shape)
        attn_weights = torch.matmul(self.linear1(x1), x2.transpose(0, 1))
        #print('attn_wights:',attn_weights.shape)
        #attn_weights = attn_weights.squeeze(dim=2)
        attn_weights = nn.functional.softmax(attn_weights, dim=1)
        
        # 使用注意力权重加权融合两个向量
        fused_x1 = torch.matmul(attn_weights, x2)
        #print('fused_x1:',fused_x1.shape)
        
        fused_x2 = torch.matmul(attn_weights.transpose(0, 1), x1)
        #print('fused_x2:',fused_x2.shape)
        return fused_x1, fused_x2
'''
#CrossAttention
class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim1, input_dim2):
        super(CrossAttentionLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim1, input_dim2)###################妙
        self.linear2 = nn.Linear(input_dim2, input_dim1)

    def forward(self, x1, x2,d1,d2):
        # 计算注意力权重
        #print('x1,x2:',x1.shape,x2.shape)
        attn_weights2 = torch.matmul(self.linear1(x1), x2.transpose(0, 1))
        #print('attn_wights:',attn_weights.shape)
        #attn_weights = attn_weights.squeeze(dim=2)
        attn_weights2 = nn.functional.softmax(attn_weights2, dim=1)
        
        # 使用注意力权重加权融合两个向量
        fused_x2 = torch.matmul(attn_weights2, x2)/np.sqrt(d2*d1)
        #fused_x1 = torch.matmul(attn_weights2, x2)/np.sqrt(d2)
        #print('fused_x1:',fused_x1.shape)
        
        attn_weights1 = torch.matmul(self.linear2(x2), x1.transpose(0, 1))
        #print('attn_wights:',attn_weights.shape)
        #attn_weights = attn_weights.squeeze(dim=2)
        attn_weights1 = nn.functional.softmax(attn_weights1, dim=1)
        fused_x1 = torch.matmul(attn_weights1, x1)/np.sqrt(d1*d2)
        #fused_x2 = torch.matmul(attn_weights1.transpose(0, 1), x1)/np.sqrt(d1)
        #print('fused_x2:',fused_x2.shape)
        return fused_x1, fused_x2
    
    
class InteractionModel_1(torch.nn.Module):
    def __init__(self, protein_model,molecular_model,pro_emb_dim,num_tasks, device,aggr = "add"):
        super( InteractionModel_1, self ).__init__() 
        self.protein_model=protein_model.to(device)
        self.alphabet=protein_model.alphabet##########.to(device)
        self.batch_converter=self.alphabet.get_batch_converter()
        self.device=device
        #self.molecular.node_representation=molecular.model.gnn()
        #self.molecular.model,self.molecular.node_representation,self.molecular.features = GNN_graphpred_1(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        #
        self.protein_sequence_representations = torch.tensor([1,2])
        #self.molecular_model############微调，必须写成类成员  self，否则就不能微调
        self.mole_model=molecular_model.gnn#nice微调，必须写成类成员  self，否则就不能微调
        self.mole_pool=molecular_model.pool#nice微调，必须写成类成员  self，否则就不能微调
        self.pred_linear = torch.nn.Linear(pro_emb_dim+molecular_model.emb_dim, num_tasks)
        self.protein_linear=torch.nn.Linear(1280,1280)
        self.protein_relu=nn.ReLU()
        
        #self.cross_attention = CrossAttentionLayer(input_dim1, input_dim2)
        self.cross_attention = CrossAttentionLayer(1280, 300)
        self.fc1 = nn.Linear(1580, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        
        
        
        
        
    #def forward(self, protein_data,*molecular_data):#******不能有，否则出错
    def forward(self, protein_data,molecular_data):#
        #print('protein_data:',protein_data.shape)
        print('molecular_data:',molecular_data.shape)
        #print('
        ############protein
        print('self.device:',self.device)
        print('protein_data:',protein_data)
            
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_data)
        batch_tokens=batch_tokens.to(self.device)######################
        print('batch_tokens:',batch_tokens.device.type)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        print('protein_batch_lens:',len(batch_lens))
        with torch.no_grad():
            results = self.protein_model(batch_tokens, repr_layers=[33], return_contacts=True)###############3
        
        protein_token_representations = results["representations"][33]
        m,n,v=protein_token_representations.shape
        print('protein_token_representations:',protein_token_representations.shape)
        for i, tokens_len in enumerate(batch_lens):
            if i==0:
                self.protein_sequence_representations=protein_token_representations[i, 1 : tokens_len - 1].mean(0)
            else:
                 self.protein_sequence_representations=torch.stack([self.protein_sequence_representations,protein_token_representations[i,1:tokens_len-1].mean(0)],dim=0)
            #self.protein_sequence_representations.append(protein_token_representations[i, 1 : tokens_len - 1].mean(0))
        self.protein_sequence_representations=self.protein_sequence_representations.reshape(-1,1280)    
        print('self.protein_sequence_representations:',self.protein_sequence_representations.shape)
        ################molecular
        self.protein_sequence_representations=self.protein_linear( self.protein_sequence_representations)############fine_tunning protein large model 
        self.protein_sequence_representations=self.protein_relu(self.protein_sequence_representations)############
        #set up optimizer
        #different learning rate for different part of GNN
        molecular_node_representation=self.mole_model(molecular_data.x,molecular_data.edge_index,molecular_data.edge_attr)
        print('molecular_node_representation:',molecular_node_representation.shape)
        molecular_representation=self.mole_pool(molecular_node_representation,molecular_data.batch)
        print('molecular_representation:',molecular_representation.shape)
        print('molecular_batch:',len(molecular_data.batch))
        #molecular_representation=self.molecular.features(*molecular_data)
        x1= self.protein_sequence_representations
        x2=molecular_representation
        fused_x1, fused_x2 =  self.cross_attention(x1, x2)  
        
        fused_x1=F.softmax(fused_x1)
        fused_x2=F.softmax(fused_x2)
        print('x1:',x1.shape)
        print('x2:',x2.shape)
        print('fused_x1:',fused_x1.shape)
        print('fused_x2:',fused_x2.shape)
        
        x11_att=x1*fused_x2
        x22_att=x2*fused_x1
        
        print('x11_att:',x11_att.shape)
        print('x22_att:',x22_att.shape)
        x11=torch.add(x11_att,x1)
        x22=torch.add(x22_att,x2)
        # 合并两个融合后的向量
        combined = torch.cat([x11, x22], dim=1)   
        #all_features=torch.concat([self.protein_sequence_representations,molecular_representation],axis=1)
        #print('all_features:',all_features.shape)
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.fc2(out)                           
        #out=self.pred_linear(all_features)
        print('out:',out)
        
        
        
        
        
        return out

class PMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,100)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(100,1)
        self.sig=nn.Sigmoid()
        self.dropout=nn.Dropout(0.5)
    def forward(x):
        out=self.dropout(self.relu1(self.fc1(x)))
        out=self.sig(self.fc2(out))
        return out
pmodel=PMLP()

class InteractionModel_4(torch.nn.Module):
    def __init__(self, protein_model,molecular_model,smile_model,protein_embd_dim,mol_embd_dim,num_tasks, device,smile_embd_dim, dropout=0.2,aggr = "mean"):
        super( InteractionModel_4, self ).__init__() 
        self.protein_embd_dim=protein_embd_dim
        self.mol_embd_dim=mol_embd_dim
        self.smile_embd_dim=smile_embd_dim
        self.protein_model=protein_model.to(device)
        self.alphabet=protein_model.alphabet##########.to(device)
        self.batch_converter=self.alphabet.get_batch_converter()
        self.device=device
        #self.molecular.node_representation=molecular.model.gnn()
        #self.molecular.model,self.molecular.node_representation,self.molecular.features = GNN_graphpred_1(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        #
        self.protein_sequence_representations = torch.tensor([1,2])
        #self.molecular_model############微调，必须写成类成员  self，否则就不能微调
        self.mole_model=molecular_model.gnn#nice微调，必须写成类成员  self，否则就不能微调
        self.mole_pool=molecular_model.pool#nice微调，必须写成类成员  self，否则就不能微调
        self.mole_linear=torch.nn.Linear(mol_embd_dim,mol_embd_dim)
        self.mol_pred_linear = torch.nn.Linear(protein_embd_dim+mol_embd_dim, num_tasks)
        self.protein_linear=torch.nn.Linear(protein_embd_dim,protein_embd_dim)
        self.protein_relu=nn.ReLU()
        self.relu=nn.ReLU()
        self.protein_embd_dim=protein_embd_dim
        #self.cross_attention = CrossAttentionLayer(input_dim1, input_dim2)
        self.mol_cross_attention = CrossAttentionLayer(protein_embd_dim, mol_embd_dim)
        self.fc1 = nn.Linear(protein_embd_dim+ mol_embd_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        
        self.desc_skip_connection = True 
        self.fcs = []  # nn.ModuleList()
        #print('dropout is {}'.format(dropout))
        
        self.smile_model=smile_model
        self.smile_linear=nn.Linear(smile_embd_dim,smile_embd_dim)
        
        self.smile_pred_linear = torch.nn.Linear(protein_embd_dim+smile_embd_dim, num_tasks)
        self.smile_cross_attention = CrossAttentionLayer(protein_embd_dim, smile_embd_dim)
        self.fc3 = nn.Linear(protein_embd_dim+ smile_embd_dim, 256)
        
        self.layer_norm_seq = nn.LayerNorm(protein_embd_dim, eps=1e-6)  # 默认对最后一个维度初始化
        self.layer_norm_mol = nn.LayerNorm(mol_embd_dim, eps=1e-6)  # 默认对最后一个维度初始化
        self.layer_norm_smile = nn.LayerNorm(smile_embd_dim, eps=1e-6)  # 默认对最后一个维度初始化
        
        
        self.fc11 = nn.Linear(smile_embd_dim, smile_embd_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        #self.fc22 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        #self.dropout2 = nn.Dropout(dropout)
        #self.relu2 = nn.GELU()
        #self.final = nn.Linear(smile_embd_dim, 1)
        #self.pmodel=PMLP().to(device)
        #self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.dropout = nn.Dropout(p=0.3)  # dropout训练
        # 定义可学习参数 t
        
        '''
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=64,kernel_size=5,stride=1),
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=5,stride=1),
            nn.AvgPool1d(5)
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=5,stride=1),
            nn.Conv1d(in_channels=64,out_channels=1,kernel_size=5,stride=1),
            nn.AvgPool1d(5)
        )
        '''
        self.sigmoid=nn.Sigmoid()
        
        self.t = nn.Parameter(torch.Tensor(1))
        self.t_linear1=nn.Linear(1,50)
        self.t_linear2=nn.Linear(50,1)
        self.t.data.fill_(0.5)  # 初始化 t 为0.5
        
    #def forward(self, protein_data,*molecular_data):#******不能有，否则出错
    def forward(self, protein_data,molecular_data,smile_data):#
        
        
            
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_data)
        batch_tokens=batch_tokens.to(self.device)######################
        #print('batch_tokens:',batch_tokens.device.type)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        #print('protein_batch_lens:',len(batch_lens),batch_lens)
        results=None
        protein_token_repressentations=None
        u=None
        with torch.no_grad():
            #results = self.protein_model(batch_tokens, repr_layers=[6], return_contacts=True)###############3
            results = self.protein_model(batch_tokens, repr_layers=[6], return_contacts=False)###############3
        protein_token_representations = results["representations"][6]
        m,n,v=protein_token_representations.shape
        #print('m,n,v:',m,n,v)
        #print('protein_token_representations:',protein_token_representations.shape)
        del results, batch_tokens
        
        
        for i, tokens_len in enumerate(batch_lens):
            if i==0:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,self.protein_embd_dim)
                self.protein_sequence_representations=u
                
            else:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,self.protein_embd_dim)
                self.protein_sequence_representations=torch.concat([self.protein_sequence_representations,u],dim=0)
                
        del protein_token_representations
        del u
        
        self.protein_sequence_representations=self.dropout(self.protein_linear( self.protein_sequence_representations))############fine_tunning protein large model 
        self.protein_sequence_representations=self.protein_relu(self.protein_sequence_representations)############
        
        
        molecular_node_reprresentation=None
        with torch.no_grad():
            molecular_node_representation=self.mole_model(molecular_data.x,molecular_data.edge_index,molecular_data.edge_attr)
            
        molecular_representation=self.mole_pool(molecular_node_representation,molecular_data.batch)
        molecular_representation=self.relu(self.dropout(self.mole_linear(molecular_representation)))
        del molecular_node_representation
        
        x1= self.protein_sequence_representations############for mol_bert
        x3=self.protein_sequence_representations######################for molformer
        x2=molecular_representation
        
        x1=self.layer_norm_seq(x1)
        x2=self.layer_norm_mol(x2)
        x3=self.layer_norm_seq(x3)
        fused_x1, fused_x2 =  self.mol_cross_attention(x1, x2,self.protein_embd_dim,self.mol_embd_dim)  
        
        
        x11_att=fused_x1
        x22_att=fused_x2
        
        x11=torch.add(x11_att,x1)
        x22=torch.add(x22_att,x2)
        del x11_att, x22_att
        
        del x1,x2,fused_x1,fused_x2
        
        #gc.collect()
        #torch.cuda.empty_cache()
        # 合并两个融合后的向量
        combined12 = torch.cat([x11, x22], dim=1)   
        
        out12 = self.fc1(combined12)
        out12 = self.dropout(self.relu(out12))
        
        out12=self.fc2(out12)
        
        
        x5,mask=smile_data
        
        length_mask=LM(mask.sum(-1))
        #print('length_mask:',length_mask.shape,length_mask)
        
        with torch.no_grad():
            token_embeddings = self.smile_model.tok_emb(x5) # each index maps to a (learnable) vector
            #print('token_embbeddings:',token_embeddings.shape)
            x6 = self.smile_model.drop(token_embeddings)
            x7 = self.smile_model.blocks(x6, length_mask=LM(mask.sum(-1)))
            #x7 = self.smile_model.blocks(x6)
            #print('smile_x7:',x7.shape)
            token_embeddings = x7
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            #print('input_mask_expanded:',input_mask_expanded.shape)
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            #print('sum_embeddings:',sum_embeddings.shape)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            #print('sum_mask:',sum_mask.shape)
            x4 = sum_embeddings / sum_mask####################################
            #print('x4:',x4.shape)
            #x4=self.smile_model(smile_data)
            x4=self.relu1(self.dropout(self.fc11(x4)))
            x4=self.layer_norm_smile(x4)
        
        fused_x3, fused_x4 =  self.smile_cross_attention(x3, x4,self.protein_embd_dim,self.smile_embd_dim)  
        
        #fused_x3=F.softmax(fused_x3)
        #fused_x4=F.softmax(fused_x4)
        #print('x3:',x3.shape)
        #print('x4:',x4.shape)
        #print('fused_x3:',fused_x3.shape)
        #print('fused_x3:',fused_x4.shape)
        
        #x33_att=x3*fused_x4
        #x44_att=x4*fused_x3
        
        x33_att=fused_x3
        x44_att=fused_x4
        #print('x33_att:',x33_att.shape)
        #print('x44_att:',x44_att.shape)
        x33=torch.add(x33_att,x3)
        x44=torch.add(x44_att,x4)
        # 合并两个融合后的向量
        combined34 = torch.cat([x33, x44], dim=1)  
        del x33_att,x44_att,fused_x3,fused_x4, x33,x44,x3,x4
        #all_features=torch.concat([self.protein_sequence_representations,molecular_representation],axis=1)
        #print('all_features:',all_features.shape)
        out34 = self.fc3(combined34)
        out34 = self.dropout(self.relu(out34))
        #out34 = torch.sigmoid(self.fc2(out34) )  
        out34=self.fc2(out34)
        #gc.collect()
        #torch.cuda.empty_cache()
        
         # 融合两个分支特征
        t1=self.relu(self.t_linear1(self.t))
        t2=torch.sigmoid(self.t_linear2(t1))
        
        out = t2 * out12 + (1 - t2) * out34
        
        
        
        return out

class InteractionModel_6(torch.nn.Module):
    def __init__(self, protein_model,molecular_model,smile_model,protein_embd_dim,mol_embd_dim,num_tasks, device,smile_embd_dim, dropout=0.2,aggr = "mean"):
        super( InteractionModel_6, self ).__init__() 
        self.protein_embd_dim=protein_embd_dim
        self.mol_embd_dim=mol_embd_dim
        self.smile_embd_dim=smile_embd_dim
        self.protein_model=protein_model.to(device)
        self.alphabet=protein_model.alphabet##########.to(device)
        self.batch_converter=self.alphabet.get_batch_converter()
        self.device=device
        #self.molecular.node_representation=molecular.model.gnn()
        #self.molecular.model,self.molecular.node_representation,self.molecular.features = GNN_graphpred_1(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        #
        self.protein_sequence_representations = torch.tensor([1,2])
        #self.molecular_model############微调，必须写成类成员  self，否则就不能微调
        self.mole_model=molecular_model.gnn#nice微调，必须写成类成员  self，否则就不能微调
        self.mole_pool=molecular_model.pool#nice微调，必须写成类成员  self，否则就不能微调
        #self.mole_linear=torch.nn.Linear(mol_embd_dim,mol_embd_dim)
        self.mole_linear=torch.nn.Linear(mol_embd_dim,256)
        self.mol_pred_linear = torch.nn.Linear(protein_embd_dim+mol_embd_dim, num_tasks)
        #self.protein_linear=torch.nn.Linear(protein_embd_dim,protein_embd_dim)
        self.protein_linear=torch.nn.Linear(protein_embd_dim,256)
        self.protein_relu=nn.ReLU()
        self.relu=nn.ReLU()
        self.protein_embd_dim=protein_embd_dim
        #self.cross_attention = CrossAttentionLayer(input_dim1, input_dim2)
        #self.mol_cross_attention = CrossAttentionLayer(protein_embd_dim, mol_embd_dim)
        self.mol_cross_attention = CrossAttentionLayer(256,256)
        self.mol_cross_attention1=CrossAttentionLayer(128,128)
        #self.fc1 = nn.Linear(protein_embd_dim+ mol_embd_dim, 256)
        #self.fc1 = nn.Linear(256+256, 256)
        
        self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(256, 1)
        self.fc2=nn.Linear(64,1)
        self.desc_skip_connection = True 
        self.fcs = []  # nn.ModuleList()
        #print('dropout is {}'.format(dropout))
        
        self.smile_model=smile_model
        #self.smile_linear=nn.Linear(smile_embd_dim,smile_embd_dim)
        self.smile_linear=nn.Linear(smile_embd_dim,256)
        self.smile_pred_linear = torch.nn.Linear(protein_embd_dim+smile_embd_dim, num_tasks)
        #self.smile_cross_attention = CrossAttentionLayer(protein_embd_dim, smile_embd_dim)
        self.smile_cross_attention = CrossAttentionLayer(256, 256)
        self.smile_cross_attention1=CrossAttentionLayer(128,128)
        #self.fc3 = nn.Linear(protein_embd_dim+ smile_embd_dim, 256)
        #self.fc3 = nn.Linear(256+256, 256)
        self.fc3=nn.Linear(64+64,64)
        #self.layer_norm_seq = nn.LayerNorm(protein_embd_dim)  # 默认对最后一个维度初始化
        #self.layer_norm_mol = nn.LayerNorm(mol_embd_dim)  # 默认对最后一个维度初始化
        #self.layer_norm_smile = nn.LayerNorm(smile_embd_dim)  # 默认对最后一个维度初始化
        self.layer_norm_seq = nn.LayerNorm(256)  # 默认对最后一个维度初始化
        self.layer_norm_mol = nn.LayerNorm(256)  # 默认对最后一个维度初始化
        self.layer_norm_smile = nn.LayerNorm(256)  # 默认对最后一个维度初始化
        
        
        self.fc88=nn.Linear(256,128)
        self.fc99=nn.Linear(128,64)
        #self.fc11 = nn.Linear(smile_embd_dim, smile_embd_dim)
        self.fc11 = nn.Linear(smile_embd_dim,256)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        #self.fc22 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        #self.dropout2 = nn.Dropout(dropout)
        #self.relu2 = nn.GELU()
        #self.final = nn.Linear(smile_embd_dim, 1)
        #self.pmodel=PMLP().to(device)
        #self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.dropout = nn.Dropout(p=0.3)  # dropout训练
        # 定义可学习参数 t
        
        '''
        self.conv1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=64,kernel_size=5,stride=1),
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=5,stride=1),
            nn.AvgPool1d(5)
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=64,kernel_size=5,stride=1),
            nn.Conv1d(in_channels=64,out_channels=1,kernel_size=5,stride=1),
            nn.AvgPool1d(5)
        )
        '''
        self.sigmoid=nn.Sigmoid()
        
        self.t = nn.Parameter(torch.Tensor(1))
        self.t_linear1=nn.Linear(1,50)
        self.t_linear2=nn.Linear(50,1)
        self.t.data.fill_(0.5)  # 初始化 t 为0.5
        
    #def forward(self, protein_data,*molecular_data):#******不能有，否则出错
    def forward(self, protein_data,molecular_data,smile_data):#
        
        
            
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_data)
        batch_tokens=batch_tokens.to(self.device)######################
        #print('batch_tokens:',batch_tokens.device.type)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        #print('protein_batch_lens:',len(batch_lens),batch_lens)
        results=None
        protein_token_repressentations=None
        u=None
        with torch.no_grad():
            #results = self.protein_model(batch_tokens, repr_layers=[6], return_contacts=True)###############3
            results = self.protein_model(batch_tokens, repr_layers=[6], return_contacts=False)###############3
        protein_token_representations = results["representations"][6]
        m,n,v=protein_token_representations.shape
        #print('m,n,v:',m,n,v)
        #print('protein_token_representations:',protein_token_representations.shape)
        del results, batch_tokens
        
        
        for i, tokens_len in enumerate(batch_lens):
            if i==0:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,self.protein_embd_dim)
                self.protein_sequence_representations=u
                
            else:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,self.protein_embd_dim)
                self.protein_sequence_representations=torch.concat([self.protein_sequence_representations,u],dim=0)
                
        del protein_token_representations
        del u
        
        self.protein_sequence_representations=self.dropout(self.protein_linear( self.protein_sequence_representations))############fine_tunning protein large model 
        self.protein_sequence_representations=self.protein_relu(self.protein_sequence_representations)############
        
        
        molecular_node_reprresentation=None
        with torch.no_grad():
            molecular_node_representation=self.mole_model(molecular_data.x,molecular_data.edge_index,molecular_data.edge_attr)
            
        molecular_representation=self.mole_pool(molecular_node_representation,molecular_data.batch)
        molecular_representation=self.relu(self.dropout(self.mole_linear(molecular_representation)))
        del molecular_node_representation
        
        x1= self.protein_sequence_representations############for mol_bert
        x3=self.protein_sequence_representations######################for molformer
        x2=molecular_representation
        
        x1=self.layer_norm_seq(x1)
        x2=self.layer_norm_mol(x2)
        x3=self.layer_norm_seq(x3)
        #fused_x1, fused_x2 =  self.mol_cross_attention(x1, x2,self.protein_embed_dim,self.mol_embd_dim)  
        fused_x1, fused_x2 =  self.mol_cross_attention(x1, x2,256,256) 
        
        x11_att=fused_x1
        x22_att=fused_x2
        
        x11=torch.add(x11_att,x1)
        x22=torch.add(x22_att,x2)
        
        x11=self.fc88(x11)
        x22=self.fc88(x22)
        
        
        fused_x1, fused_x2 =  self.mol_cross_attention1(x11, x22,128,128) 
        
        x111_att=fused_x1
        x222_att=fused_x2
        
        x111=torch.add(x111_att,x11)
        x222=torch.add(x222_att,x22)
        x111=self.fc99(x111)
        x222=self.fc99(x222)
        
        del x11_att, x22_att
        del x1,x2,fused_x1,fused_x2
        
        del x111_att, x222_att
        del x11,x22
        #gc.collect()
        #torch.cuda.empty_cache()
        # 合并两个融合后的向量
        combined12 = torch.cat([x111, x222], dim=1)   
        
        out12 = self.fc3(combined12)
        out12 = self.dropout(self.relu(out12))
        
        out12=self.fc2(out12)
        
        
        x5,mask=smile_data
        
        length_mask=LM(mask.sum(-1))
        #print('length_mask:',length_mask.shape,length_mask)
        
        with torch.no_grad():
            token_embeddings = self.smile_model.tok_emb(x5) # each index maps to a (learnable) vector
            #print('token_embbeddings:',token_embeddings.shape)
            x6 = self.smile_model.drop(token_embeddings)
            x7 = self.smile_model.blocks(x6, length_mask=LM(mask.sum(-1)))
            #x7 = self.smile_model.blocks(x6)
            #print('smile_x7:',x7.shape)
            token_embeddings = x7
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            #print('input_mask_expanded:',input_mask_expanded.shape)
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            #print('sum_embeddings:',sum_embeddings.shape)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            #print('sum_mask:',sum_mask.shape)
            x4 = sum_embeddings / sum_mask####################################
            #print('x4:',x4.shape)
            #x4=self.smile_model(smile_data)
            x4=self.relu1(self.dropout(self.fc11(x4)))
            x4=self.layer_norm_smile(x4)
        
        #fused_x3, fused_x4 =  self.smile_cross_attention(x3, x4,self.protein_embd_dim,self.smile_embd_dim)  
        fused_x3, fused_x4 =  self.smile_cross_attention(x3, x4,256,256)  
        #fused_x3=F.softmax(fused_x3)
        #fused_x4=F.softmax(fused_x4)
        #print('x3:',x3.shape)
        #print('x4:',x4.shape)
        #print('fused_x3:',fused_x3.shape)
        #print('fused_x3:',fused_x4.shape)
        
        #x33_att=x3*fused_x4
        #x44_att=x4*fused_x3
        
        x33_att=fused_x3
        x44_att=fused_x4
        #print('x33_att:',x33_att.shape)
        #print('x44_att:',x44_att.shape)
        x33=torch.add(x33_att,x3)
        x44=torch.add(x44_att,x4)
        x33=self.fc88(x33)
        x44=self.fc88(x44)
        
        fused_x3, fused_x4 =  self.smile_cross_attention1(x33, x44,128,128)  
        
        x333_att=fused_x3
        x444_att=fused_x4
        #print('x33_att:',x33_att.shape)
        #print('x44_att:',x44_att.shape)
        x333=torch.add(x333_att,x33)
        x444=torch.add(x444_att,x44)
        x333=self.fc99(x333)
        x444=self.fc99(x444)
        del x33_att,x44_att,fused_x3,fused_x4, x33,x44,x3,x4, x333_att,x444_att
        
        # 合并两个融合后的向量
        combined34 = torch.cat([x333, x444], dim=1)  
        #del x33_att,x44_att,fused_x3,fused_x4, x33,x44,x3,x4
        #all_features=torch.concat([self.protein_sequence_representations,molecular_representation],axis=1)
        #print('all_features:',all_features.shape)
        out34 = self.fc3(combined34)
        out34 = self.dropout(self.relu(out34))
        #out34 = torch.sigmoid(self.fc2(out34) )  
        out34=self.fc2(out34)
        #gc.collect()
        #torch.cuda.empty_cache()
        
         # 融合两个分支特征
        t1=self.relu(self.t_linear1(self.t))
        t2=torch.sigmoid(self.t_linear2(t1))
        
        out = t2 * out12 + (1 - t2) * out34
        
        
        
        return out
    
    
    
    
class InteractionModel_3(torch.nn.Module):
    def __init__(self, protein_model,molecular_model,smile_model,protein_embd_dim,mol_embd_dim,num_tasks, device,smile_embd_dim, dropout=0.2,aggr = "add"):
        super( InteractionModel_3, self ).__init__() 
        self.protein_model=protein_model.to(device)
        self.alphabet=protein_model.alphabet##########.to(device)
        self.batch_converter=self.alphabet.get_batch_converter()
        self.device=device
        #self.molecular.node_representation=molecular.model.gnn()
        #self.molecular.model,self.molecular.node_representation,self.molecular.features = GNN_graphpred_1(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        #
        self.protein_sequence_representations = torch.tensor([1,2])
        #self.molecular_model############微调，必须写成类成员  self，否则就不能微调
        self.mole_model=molecular_model.gnn#nice微调，必须写成类成员  self，否则就不能微调
        self.mole_pool=molecular_model.pool#nice微调，必须写成类成员  self，否则就不能微调
        self.mol_pred_linear = torch.nn.Linear(protein_embd_dim+mol_embd_dim, num_tasks)
        self.protein_linear=torch.nn.Linear(protein_embd_dim,protein_embd_dim)
        self.protein_relu=nn.ReLU()
        self.protein_embd_dim=protein_embd_dim
        #self.cross_attention = CrossAttentionLayer(input_dim1, input_dim2)
        self.mol_cross_attention = CrossAttentionLayer(protein_embd_dim, mol_embd_dim)
        self.fc1 = nn.Linear(protein_embd_dim+ mol_embd_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        
        self.desc_skip_connection = True 
        self.fcs = []  # nn.ModuleList()
        #print('dropout is {}'.format(dropout))
        
        self.smile_model=smile_model
        
        
        self.smile_pred_linear = torch.nn.Linear(protein_embd_dim+smile_embd_dim, num_tasks)
        self.smile_cross_attention = CrossAttentionLayer(protein_embd_dim, smile_embd_dim)
        self.fc3 = nn.Linear(protein_embd_dim+ smile_embd_dim, 256)
        
        
        
        self.fc11 = nn.Linear(smile_embd_dim, smile_embd_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        #self.fc22 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
        #self.dropout2 = nn.Dropout(dropout)
        #self.relu2 = nn.GELU()
        #self.final = nn.Linear(smile_embd_dim, 1)
        self.pmodel=PMLP().to(device)
        
        
    #def forward(self, protein_data,*molecular_data):#******不能有，否则出错
    def forward(self, protein_data,molecular_data,smile_data):#
        ############protein
        #SSprint('self.device:',self.device)
        #print('protein_data:',protein_data)
        #print('potein_data,molecular_data,smile_data:',molecular_data.shape,smile_data.shape)    
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_data)
        batch_tokens=batch_tokens.to(self.device)######################
        #print('batch_tokens:',batch_tokens.device.type)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        print('protein_batch_lens:',len(batch_lens),batch_lens)
        with torch.no_grad():
            results = self.protein_model(batch_tokens, repr_layers=[33], return_contacts=True)###############3
        
        protein_token_representations = results["representations"][33]
        m,n,v=protein_token_representations.shape
        print('m,n,v:',m,n,v)
        #print('protein_token_representations:',protein_token_representations.shape)
        
        
        for i, tokens_len in enumerate(batch_lens):
            if i==0:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,self.protein_embd_dim)
                #u=protein_token_representations[i, 1 : tokens_len - 1].mean(0)############why 1,not 0
                #self.protein_sequence_representations=protein_token_representations[i, 1 : tokens_len - 1].mean(0)
                self.protein_sequence_representations=u
                print('self.protein_sequence_representations:',self.protein_sequence_representations.shape)
                #self.protein_sequence_representations.reshape(-1,self.protein_embd_dim)
            else:
                u=protein_token_representations[i, 1 : tokens_len - 1].mean(0).reshape(-1,self.protein_embd_dim)
                #u=protein_token_representations[i, 1 : tokens_len - 1].mean(0)
                                #self.protein_sequence_representations=torch.stack([self.protein_sequence_representations,protein_token_representations[i,1:tokens_len-1].mean(0)],dim=1)###############dim=0,dim=1
                self.protein_sequence_representations=torch.concat([self.protein_sequence_representations,u],dim=0)
                            
                #print('self.protein_sequence_representations:',self.protein_sequence_representations.shape)
                #self.protein_sequence_representations.append(protein_token_representations[i, 1 : tokens_len - 1].mean(0))
                #print('self.protein_sequence_representations:',self.protein_sequence_representations.shape)
                #print('self.protein_sequence_representations:',self.protein_sequence_representations.shape)
            
        ################molecular
        
        #self.protein_sequence_representations=protein_token_representations[:,1:tokens_len-1].mean(0)
        print('self.protein_sequence_representations:',self.protein_sequence_representations.shape)
        self.protein_sequence_representations=self.protein_linear( self.protein_sequence_representations)############fine_tunning protein large model 
        self.protein_sequence_representations=self.protein_relu(self.protein_sequence_representations)############
        #set up optimizer
        #different learning rate for different part of GNN
        molecular_node_representation=self.mole_model(molecular_data.x,molecular_data.edge_index,molecular_data.edge_attr)
        #print('molecular_node_representation:',molecular_node_representation.shape)
        molecular_representation=self.mole_pool(molecular_node_representation,molecular_data.batch)
        #print('molecular_representation:',molecular_representation.shape)
        #print('molecular_batch:',len(molecular_data.batch))
        #molecular_representation=self.molecular.features(*molecular_data)
        x1= self.protein_sequence_representations############for mol_bert
        x3=self.protein_sequence_representations######################for molformer
        x2=molecular_representation
        fused_x1, fused_x2 =  self.mol_cross_attention(x1, x2)  
        
        fused_x1=F.softmax(fused_x1)
        fused_x2=F.softmax(fused_x2)
        #print('x1:',x1.shape)
        #print('x2:',x2.shape)
        #print('fused_x1:',fused_x1.shape)
        #print('fused_x2:',fused_x2.shape)
        
        x11_att=x1*fused_x2
        x22_att=x2*fused_x1
        
        #print('x11_att:',x11_att.shape)
        #print('x22_att:',x22_att.shape)
        x11=torch.add(x11_att,x1)
        x22=torch.add(x22_att,x2)
        # 合并两个融合后的向量
        combined12 = torch.cat([x11, x22], dim=1)   
        #all_features=torch.concat([self.protein_sequence_representations,molecular_representation],axis=1)
        #print('all_features:',all_features.shape)
        out12 = self.fc1(combined12)
        out12 = self.relu(out12)
        out12 = self.fc2(out12)                           
        #out=self.pred_linear(all_features)
        print('out12:',out12)
        
        x5,mask=smile_data
        print('x5:',x5.shape)
        print('mask:',mask.shape)
        length_mask=LM(mask.sum(-1))
        print('length_mask:',length_mask.shape)
        token_embeddings = self.smile_model.tok_emb(x5) # each index maps to a (learnable) vector
        print('token_embbeddings:',token_embeddings.shape)
        x6 = self.smile_model.drop(token_embeddings)
        #x7 = self.smile_model.blocks(x6, length_mask=LM(mask.sum(-1)))
        x7 = self.smile_model.blocks(x6)
        token_embeddings = x7
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        print('input_mask_expanded:',input_mask_expanded.shape)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        print('sum_embeddings:',sum_embeddings.shape)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        print('sum_mask:',sum_mask.shape)
        x4 = sum_embeddings / sum_mask####################################
        print('x4:',x4.shape)
        #x4=self.smile_model(smile_data)
        x4=self.relu1(self.fc11(x4))
        
        fused_x3, fused_x4 =  self.smile_cross_attention(x3, x4)  
        
        fused_x3=F.softmax(fused_x3)
        fused_x4=F.softmax(fused_x4)
        #print('x3:',x3.shape)
        #print('x4:',x4.shape)
        #print('fused_x3:',fused_x3.shape)
        #print('fused_x3:',fused_x4.shape)
        
        x33_att=x3*fused_x4
        x44_att=x4*fused_x3
        
        #print('x33_att:',x33_att.shape)
        #print('x44_att:',x44_att.shape)
        x33=torch.add(x33_att,x3)
        x44=torch.add(x44_att,x4)
        # 合并两个融合后的向量
        combined34 = torch.cat([x33, x44], dim=1)   
        #all_features=torch.concat([self.protein_sequence_representations,molecular_representation],axis=1)
        #print('all_features:',all_features.shape)
        out34 = self.fc3(combined34)
        out34 = self.relu(out34)
        out34 = self.fc2(out34)   
        
        #print('out12:',out12.shape,out12)
        u=self.pmodel(out12)
        #print('u:',u.shape,u)
        out=u*out12+(1-u)*out34
        
        #out=out12*0.5+out34*0.5
        
        
        
        
        return out




# In[ ]:
class SequenceModel(torch.nn.Module):
    def __init__(self):
        super( SequenceModel, self ).__init__() 
        self.protein_model, self.protein_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter=self.protein_alphabet.get_batch_converter()
        
        
        self.protein_sequence_representations = []
        
        
    def forward(self, protein_data):
        ############protein
        print('protein_data:',protein_data)
            
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = protein_model(batch_tokens, repr_layers=[33], return_contacts=True)
        
        protein_token_representations = results["representations"][33]
        print('protein_token_representations:',protein_token_representations)
        for i, tokens_len in enumerate(batch_lens):
            self.protein_sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            
        
        return self.protein_sequence_representations







# In[ ]:






