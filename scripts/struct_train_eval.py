import csv
import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch import optim

from read_data import read_data_structures
from utils import normalize_adjacency, sparse_mx_to_torch_sparse_tensor
from struct_model import GNN
from write_data import write_sub

# Load graphs
adj, features, edge_features = read_data_structures()

# Normalize adjacency matrices
adj = [normalize_adjacency(A) for A in adj]

# Split data into training and test sets
adj_train = list()
features_train = list()
y_train = list()
adj_test = list()
features_test = list()
proteins_test = list()
with open("graph_labels.txt", "r") as f:
    for i, line in enumerate(f):
        t = line.split(",")
        if len(t[1][:-1]) == 0:
            proteins_test.append(t[0])
            adj_test.append(adj[i])
            features_test.append(features[i])
        else:
            adj_train.append(adj[i])
            features_train.append(features[i])
            y_train.append(int(t[1][:-1]))

# Initialize device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 50
batch_size = 64
n_hidden = 64
n_input = 86
dropout = 0.2
learning_rate = 0.001
n_class = 18

# Compute number of training and test samples
N_train = len(adj_train)
N_test = len(adj_test)

# Initializes model and optimizer
model = GNN(n_input, n_hidden, dropout, n_class).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Train model
for epoch in range(epochs):
    t = time.time()
    model.train()
    train_loss = 0
    correct = 0
    count = 0
    # Iterate over the batches
    for i in range(0, N_train, batch_size):
        adj_batch = list()
        features_batch = list()
        idx_batch = list()
        y_batch = list()

        # Create tensors
        for j in range(i, min(N_train, i + batch_size)):
            n = adj_train[j].shape[0]
            adj_batch.append(adj_train[j] + sp.identity(n))
            features_batch.append(features_train[j])
            idx_batch.extend([j - i] * n)
            y_batch.append(y_train[j])

        adj_batch = sp.block_diag(adj_batch)
        features_batch = np.vstack(features_batch)

        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
        features_batch = torch.FloatTensor(features_batch).to(device)
        idx_batch = torch.LongTensor(idx_batch).to(device)
        y_batch = torch.LongTensor(y_batch).to(device)

        optimizer.zero_grad()
        output = model(features_batch, adj_batch, idx_batch)
        loss = loss_function(output, y_batch)
        train_loss += loss.item() * output.size(0)
        count += output.size(0)
        preds = output.max(1)[1].type_as(y_batch)
        correct += torch.sum(preds.eq(y_batch).double())
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(
            "Epoch: {:03d}".format(epoch + 1),
            "loss_train: {:.4f}".format(train_loss / count),
            "acc_train: {:.4f}".format(correct / count),
            "time: {:.4f}s".format(time.time() - t),
        )

# Evaluate model
model.eval()
y_pred_proba = list()
# Iterate over the batches
for i in range(0, N_test, batch_size):
    adj_batch = list()
    idx_batch = list()
    features_batch = list()
    y_batch = list()

    # Create tensors
    for j in range(i, min(N_test, i + batch_size)):
        n = adj_test[j].shape[0]
        adj_batch.append(adj_test[j] + sp.identity(n))
        features_batch.append(features_test[j])
        idx_batch.extend([j - i] * n)

    adj_batch = sp.block_diag(adj_batch)
    features_batch = np.vstack(features_batch)

    adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
    features_batch = torch.FloatTensor(features_batch).to(device)
    idx_batch = torch.LongTensor(idx_batch).to(device)

    output = model(features_batch, adj_batch, idx_batch)
    y_pred_proba.append(output)

y_pred_proba = torch.cat(y_pred_proba, dim=0)
y_pred_proba = torch.exp(y_pred_proba)
y_pred_proba = y_pred_proba.detach().cpu().numpy()


write_sub(y_pred_proba, proteins_test)
