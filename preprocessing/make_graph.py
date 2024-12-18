# -*- coding: utf-8 -*-
"""
Created on Wed Dec  18 09:47:16 2024
Create the heterogenous snoRNA-disease graph
@author: Massimo La Rosa
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import normalize

outdir = "../data/graph_data/"
indir = "../data/raw_data"

#function to create node mapping
def get_node(feature_mat):
    mapping = {index: i for i, index in enumerate(feature_mat.index)}
    return mapping

#function to create edges bewteen snoRNA and disease
def get_edge(src_index,dst_index,src_mapping,dst_mapping,df):
    src = [src_mapping[index] for index in df[src_index]]
    dst = [dst_mapping[index] for index in df[dst_index]]
    edge_index = torch.tensor([src,dst])

    return edge_index

#function to create weighted edges bewteen snoRNA and disease
def get_edge_w(src_index,dst_index,src_mapping,dst_mapping,df):
    src = [src_mapping[index] for index in df[src_index]]
    dst = [dst_mapping[index] for index in df[dst_index]]
    edge_index = torch.tensor([src,dst])
    edge_weight = torch.tensor(df['score'].values)
    
    return edge_index, edge_weight
    

#load snoRNA-disease associations
sda = pd.read_csv(indir+'sd_association.csv', index_col = 0,  sep=',')
sda = sda.reset_index(drop=True)
sda = sda[sda['DO_ID'].str.contains('DOID')]

#load disease features
df_disease = pd.read_csv(indir+"disease_feat_bgeicl.csv",index_col=0, sep=",")

#load snoRNA features
snorna_feat = pd.read_csv(indir+'snorna_feature_2-4.csv', index_col=0)
snorna_feat = snorna_feat.T
snorna_feat = snorna_feat.reindex(index=sda['Symbol'].unique())

#L2 normalize snorna features
snorna_norm = normalize(snorna_feat)

df_snorna = pd.DataFrame(snorna_norm)
df_snorna.set_index(snorna_feat.index, inplace=True)

#compute snorna and disease node mapping
snorna_mapping = get_node(df_snorna)
disease_mapping = get_node(df_disease)

#create empty heterogenous graph
data = HeteroData()

#compute node features
data['snorna'].node_id = torch.arange(len(sda['Symbol'].unique()))
data['snorna'].x = torch.tensor(df_snorna.to_numpy())
data['disease'].x = torch.tensor(df_disease.to_numpy())
data['disease'].node_id = torch.arange(len(sda['DO_ID'].unique()))

#compute edges
edge_index = get_edge('Symbol','DO_ID',snorna_mapping,disease_mapping,sda)
data['snorna','disease'].edge_index = edge_index

#compute edges with weights
edge_index, edge_weight = get_edge_w('Symbol','DO_ID',snorna_mapping,disease_mapping,sda)
data['snorna','disease'].edge_index = edge_index
data['snorna','disease'].edge_weight = edge_weight

#save the graph
torch.save(data,outdir+"hetero_graph_2-4.pkl")
















