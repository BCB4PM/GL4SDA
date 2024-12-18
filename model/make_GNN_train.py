# -*- coding: utf-8 -*-
"""
Created on Wed Dec  18 10:40:53 2024
Train and validate the GNN model
@author: Massimo La Rosa
"""

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, GraphConv, HeteroConv
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything
import random 
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score

traindir = "../data/graph_data/train/" # directory with the training graph
result_dir = "../result/" # directory for saving results
input_graph = "hetero_graph_2-4_train.pkl" # training graph input file

seed_model = 35 #seed for model training
seed = 41 # seed for train/validation splitting

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


# Class that implements the GNN with GraphConv and GATv2Conv operators, with edge weights
class HeterowGraphGNN(torch.nn.Module):
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


    def forward(self, data, x_dict, edge_index, edge_weight):
        
        x_dict = self.convs[0](x_dict, edge_index, edge_weight)
        x_dict = {key: z.relu() for key, z in x_dict.items()}
        x_dict = {key: F.dropout(z, p=self.p, training=self.training) for key, z in x_dict.items()}
 
        x_dict = self.convs[1](x_dict, edge_index)
        x_dict = {key: z.relu() for key, z in x_dict.items()}
        x_dict = {key: F.dropout(z, p=self.p, training=self.training) for key, z in x_dict.items()}
        
        x_dict = self.convs[2](x_dict, edge_index, edge_weight)
        
        pred = self.classifier(
            x_dict['snorna'],
            x_dict['disease'],
            data['snorna','to','disease'].edge_label_index
        )
        
        return pred 

# Class that implements the GNN with SAGEConv and GATv2Conv operators
class HeteroSAGEGNN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_rate=0.4, num_heads=8):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()

        conv1 = HeteroConv({
            ('snorna', 'to', 'disease'): SAGEConv((-1, -1), hidden_channels[0], normalize=True),
            ('disease', 'rev_to', 'snorna'): SAGEConv((-1, -1), hidden_channels[0], normalize=True),
        }, aggr='sum')        
        self.convs.append(conv1)
        
        conv2 = HeteroConv({
              ('snorna', 'to', 'disease'): GATv2Conv((-1, -1), hidden_channels[1], add_self_loops=False, heads=num_heads, concat=False),
              ('disease', 'rev_to', 'snorna'): GATv2Conv((-1, -1), hidden_channels[1], add_self_loops=False, heads=num_heads, concat=False),
        }, aggr='sum')
        self.convs.append(conv2)
        
        
        conv3 = HeteroConv({
            ('snorna', 'to', 'disease'): SAGEConv((-1, -1), hidden_channels[2], normalize=True),
            ('disease', 'rev_to', 'snorna'): SAGEConv((-1, -1), hidden_channels[2], normalize=True),
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



n_epochs = 500

hidden_1 = 128
hidden_2 = 128
hidden_3 = 64
hidden_channels = [hidden_1,hidden_2,hidden_3]

lr = 0.0001
criterion = torch.nn.BCEWithLogitsLoss() # Choose loss function. We are working directly with logits, i.e. it makes the sigmoid first.

# load train graph
graph = torch.load(traindir+input_graph)

#set the seed for reproducibility
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
    disjoint_train_ratio=0.2, # ratio between supervision and message passing edges
    neg_sampling_ratio=1.0,
    add_negative_train_samples=False,
    edge_types=("snorna", "to", "disease"),
    rev_edge_types=("disease", "rev_to", "snorna"), 
    )

train_data, val_data, empty_data = transform(graph)

# Define train seed edges and add negative samples to the training set:
edge_label_index = train_data["snorna", "to", "disease"].edge_label_index
edge_label = train_data["snorna", "to", "disease"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 15],
    neg_sampling_ratio=2.0,
    edge_label_index=(("snorna", "to", "disease"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=False,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set the seed for the GNN model
seed_everything(seed_model)

#instantiate the model. Choose the model among HeteroGraphGNN, HeterowGraphGNN and HeteroSAGEGNN
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
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")


######################################VALIDATION###############################
# We do not need to add negative samples for the validation set. It already has them.

# Define validation seed edges:
edge_label_index = val_data["snorna", "to", "disease"].edge_label_index
edge_label = val_data["snorna", "to", "disease"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[20, 15],
    edge_label_index=(("snorna", "to", "disease"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=False,
)

preds = []
ground_truths = []
with torch.no_grad():
    model.eval()
    for sampled_val_data in val_loader:
        sampled_val_data.to(device)
        preds.append(model(sampled_val_data,sampled_val_data.x_dict,sampled_val_data.edge_index_dict))
        ground_truths.append(sampled_val_data["snorna", "to", "disease"].edge_label)
pred_row = torch.cat(preds, dim=0).cpu() 
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_row = torch.cat(ground_truths, dim=0).cpu()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc_val = roc_auc_score(ground_truth, pred)
acc_val = accuracy_score( ground_truth, pred_row.sigmoid().round() )
val_pre = precision_score(ground_truth,pred_row.sigmoid().round())
val_rec = recall_score(ground_truth,pred_row.sigmoid().round())
val_mcc = matthews_corrcoef(ground_truth,pred_row.sigmoid().round())
val_f1 = f1_score(ground_truth,pred_row.sigmoid().round())

val_file = result_dir+"validation_results.csv"
foutput = open(val_file,'w')
foutput.write("Input_graph,epochs,hidden1,hidden2,hidden3,Accuracy,Precision,Recall,F1_score,MCC,AUC\n")
    
foutput.write(input_graph.split(".")[0]+","+str(n_epochs)+","+str(hidden_channels[0])+","+str(hidden_channels[1])+","+str(hidden_channels[2])+","
            +str(acc_val)+","+str(val_pre)+","+str(val_rec)+","+str(val_f1)+","+str(val_mcc)+","+str(auc_val)+"\n")

foutput.close()


print()
print(f"Validation AUC: {auc_val:.4f}")
print(f"Validation ACC: {acc_val:.4f}")
print(f"Validation Prec: {val_pre:.4f}")
print(f"Validation Recall: {val_rec:.4f}")
print(f"Validation MCC: {val_mcc:.4f}")
print(f"Validation F-1: {val_f1:.4f}")