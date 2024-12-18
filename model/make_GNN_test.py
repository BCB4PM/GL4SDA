# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:39:33 2024
Train and test the GNN model
@author: Massimo La Rosa
"""

import pandas as pd
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv, GraphConv, HeteroConv
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import random 
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score, roc_curve
from torch_geometric import seed_everything

traindir = "../data/graph_data/train/" # directory with the training graph
testdir = "../data/graph_data/test/" # directory with the test graph
resultdir = "../result/" # directory for saving results
input_graph = "hetero_graph_2-4_train.pkl" # training graph input file
test_graph = "hetero_graph_2-4_test.pkl" # test graph input file

graph = torch.load(traindir+input_graph)
test_data = torch.load(testdir+test_graph)

seed_model = 35 #seed for model training
seed = 41 # seed for train/validation splitting

# function that merges the validation supervision edges into the training supervision edges
def merge_edge(train_graph,val_graph):
    
    rev_graph = copy.copy(train_graph)
    val_label = val_graph['snorna','to','disease'].edge_label.detach().numpy()
    val_label = pd.DataFrame(val_label)

    z = val_label[val_label[0] == 1.0]
    
    val_index = val_graph['snorna','to','disease'].edge_label_index.detach().numpy()
    val_index = pd.DataFrame(val_index)

    v = val_index.iloc[:,z.index]
    v = torch.tensor(v.values)

    rev_graph['snorna','to','disease'].edge_label_index = torch.cat((rev_graph['snorna','to','disease'].edge_label_index,v),dim=1)
    
    z = torch.tensor(z.values)
    z = torch.squeeze(z)
    
    rev_graph['snorna','to','disease'].edge_label = torch.cat((rev_graph['snorna','to','disease'].edge_label,z))

    return rev_graph

# set the seed for reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)            # if you are using multi-GPU.
np.random.seed(seed)                        # Numpy module.
random.seed(seed)                           # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# split train and validation set and add negative samples to the validation set
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.0,
    is_undirected = True,
    disjoint_train_ratio=0.2,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=False,
    edge_types=("snorna", "to", "disease"),
    rev_edge_types=("disease", "rev_to", "snorna"), 
    )

train_data, val_data, empty_data = transform(graph)

# add validation supervision edges to training supervision edges, that are of course disjoint from training message passing edges
train_data2 = merge_edge(train_data, val_data)

# Define train seed edges and add negative samples to the training set:
edge_label_index = train_data2["snorna", "to", "disease"].edge_label_index
edge_label = train_data2["snorna", "to", "disease"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data2,
    num_neighbors=[20, 15],
    neg_sampling_ratio=2.0,
    edge_label_index=(("snorna", "to", "disease"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=False,
)

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


# Class that implements the GNN with GraphConv and GATv2Conv operators, without edge weights
class HeteroGraphGNN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_rate=0.4, num_heads=8):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()

        conv1 = HeteroConv({
            ('snorna', 'to', 'disease'): GraphConv((-1, -1), hidden_channels[0]),
            ('disease', 'rev_to', 'snorna'): GraphConv((-1, -1), hidden_channels[0]),
        }, aggr='sum')        
        self.convs.append(conv1)
        
        conv2 = HeteroConv({
              ('snorna', 'to', 'disease'): GATv2Conv((-1, -1), hidden_channels[1], add_self_loops=False, heads=num_heads, concat=False),
              ('disease', 'rev_to', 'snorna'): GATv2Conv((-1, -1), hidden_channels[1], add_self_loops=False, heads=num_heads, concat=False),
        }, aggr='sum')
        self.convs.append(conv2)
        
        
        conv3 = HeteroConv({
            ('snorna', 'to', 'disease'): GraphConv((-1, -1), hidden_channels[2]),
            ('disease', 'rev_to', 'snorna'): GraphConv((-1, -1), hidden_channels[2]),
        }, aggr='sum')        
        self.convs.append(conv3)
        
    
        self.p = dropout_rate

        self.classifier = Classifier()


    def forward(self, data, x_dict, edge_index):
        
 
        for cc, conv in enumerate(self.convs):
           
            x_dict = conv(x_dict, edge_index)

            if cc != (len(self.convs)-1):
                x_dict = {key: z.relu() for key, z in x_dict.items()}
                x_dict = {key: F.dropout(z, p=self.p, training=self.training) for key, z in x_dict.items()}

        pred = self.classifier(
            x_dict['snorna'],
            x_dict['disease'],
            data['snorna','to','disease'].edge_label_index
        )
        
        return pred 

hidden_1 = 128
hidden_2 = 128
hidden_3 = 64
hidden_channels = [hidden_1,hidden_2,hidden_3]
lr = 0.0001
criterion = torch.nn.BCEWithLogitsLoss() # Choose loss function. We are working directly with logits, i.e. it makes the sigmoid first.
n_epochs = 500 # This obviously depends on the lr. You might want to increase this if lr is small.


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set the seed for the GNN model
seed_everything(seed_model)

#instantiate the model.
model = HeteroGraphGNN(hidden_channels=hidden_channels)
model = model.double()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Choose optimizer.

# start training
for epoch in range(n_epochs):
    model.train()
    total_loss = total_examples = 0
    for sampled_data in train_loader:
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data,sampled_data.x_dict,sampled_data.edge_index_dict)
        ground_truth = sampled_data["snorna", "to", "disease"].edge_label
        loss = criterion(pred, ground_truth)
        loss.backward()
        optimizer.step()

# Define test seed edges:
edge_label_index = test_data["snorna", "to", "disease"].edge_label_index
edge_label = test_data["snorna", "to", "disease"].edge_label
test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[20, 15],
    edge_label_index=(("snorna", "to", "disease"), edge_label_index),
    edge_label=edge_label,
    batch_size=128, #64,
    shuffle=False,
)


######################################TEST###############################
preds = []
ground_truths = []
for sampled_test_data in test_loader:
    with torch.no_grad():
        sampled_test_data.to(device)
        preds.append(model(sampled_test_data,sampled_test_data.x_dict,sampled_test_data.edge_index_dict))
        ground_truths.append(sampled_test_data["snorna", "to", "disease"].edge_label)
pred_row = torch.cat(preds, dim=0).cpu()       
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth_row = torch.cat(ground_truths, dim=0).cpu()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc_test = roc_auc_score(ground_truth, pred)
acc_test = accuracy_score( ground_truth, pred_row.sigmoid().round() )
test_pre = precision_score(ground_truth,pred_row.sigmoid().round())
test_rec = recall_score(ground_truth,pred_row.sigmoid().round())
test_mcc = matthews_corrcoef(ground_truth,pred_row.sigmoid().round())
test_f1 = f1_score(ground_truth,pred_row.sigmoid().round())
fpr, tpr, _ = roc_curve(ground_truth,pred)

test_file = resultdir+"test_results.csv"
foutput = open(test_file,'w')
foutput.write("Input_graph,epochs,hidden1,hidden2,hidden3,Accuracy,Precision,Recall,F1_score,MCC,AUC\n")
    
foutput.write(input_graph.split(".")[0]+","+str(n_epochs)+","+str(hidden_channels[0])+","+str(hidden_channels[1])+","+str(hidden_channels[2])+","
            +str(acc_test)+","+str(test_pre)+","+str(test_rec)+","+str(test_f1)+","+str(test_mcc)+","+str(auc_test)+"\n")

foutput.close()


print()
print(f"Test AUC: {auc_test:.4f}")
print(f"Test ACC: {acc_test:.4f}")
print(f"Test Prec: {test_pre:.4f}")
print(f"Test Recall: {test_rec:.4f}")
print(f"Test MCC: {test_mcc:.4f}")
print(f"Test F-1: {test_f1:.4f}")


