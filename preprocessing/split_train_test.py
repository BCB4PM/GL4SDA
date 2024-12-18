# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:03:39 2024
Create train and test graphs
@author: Massimo La Rosa
"""

import numpy as np
import torch
import torch_geometric.transforms as T
import random 

outtrain = "../data/graph_data/train/"
outtest = "../data/graph_data/test/"
graphdir = "../data/graph_data/"
input_graph = "hetero_graph_2-4.pkl"

#load the graph
graph = torch.load(graphdir+input_graph)
graph = T.ToUndirected()(graph)

#set the seed for reproducibility
seed = 35
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)            # if you are using multi-GPU.
np.random.seed(seed)                        # Numpy module.
random.seed(seed)                           # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#split train and test set and add negative samples to the test set
transform = T.RandomLinkSplit(
    num_val=0.0,
    num_test=0.1,
    is_undirected = True,
    disjoint_train_ratio=0.0,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=False,
    edge_types=("snorna", "to", "disease"),
    rev_edge_types=("disease", "rev_to", "snorna"), 
    )

train_data, val_data, test_data = transform(graph)

train_data['snorna','to','disease']['edge_label']= None
train_data['snorna','to','disease']['edge_label_index']= None

#save train and test graphs
torch.save(train_data,outtrain+input_graph.split(".")[0]+"_train.pkl")
torch.save(test_data,outtest+input_graph.split(".")[0]+"_test.pkl")
