# GL4SDA
A GNN model based on snoRNA secondary structures and LLM disease embedding for the prediction of snoRNA-disease associations.

This repository provides the source code for "insert the paper's title".

- The "**data**" folder contains raw and processed data. The "raw_data" sub-folder contains siRNA sequences, disease descriptions, as well as their feature representations, and 911 snoRNA-disease associations. Otherwise, the "graph_data" sub-folder contains a graph representation of soRNA-disease associations and two subgraphs representing the train and the test sets.

- The "**preprocessing**" folder contains two scripts we released for transforming raw data into the inputs of our model.

- The "**model**" folder contains the main scripts for predicting and testing siRNA-disease associations. The results of these scripts can be found in the "result" folder.


## Requirements and Dependencies 

To write and run our Python scripts, we used Python= 3.10.

The other essential packages are:

pyg= 2.6

pytorch= 2.4.1

scikit-learn = 1.3.2

pandas = 2.1.3

numpy = 1.26.2


## Usage

We provide a train and a test graph for a snoRNA-disease association prediction.
Please run the scripts in the "model" folder with our prepared dataset in the following order:

```
$ python make_GNN_train.py
$ python make_GNN_test.py
```


We also provide all the row data necessary to create train and test graphs.
Please run the scripts in the "preprocessing" folder in the following order:

```
$ python make_graph.py
$ python split_train_test.py
```

