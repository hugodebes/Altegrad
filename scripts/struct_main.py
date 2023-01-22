# basics
import numpy as np


# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from models.deep_learning.struct_model import GNN

from sklearn.preprocessing import StandardScaler
from struct_train_eval import predict, train

from utils.read_data import read_data_structures, read_list
from utils.struct_utils import (
    normalize_adjacency,
    replace_ft,
    resize_prot_seq,
    split_data,
)
from utils.write_data import write_sub

sequence_embeddings_path = ""
graph_labels_path = "data/graph_labels.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    adj, features, edge_features = read_data_structures()
    adj = [normalize_adjacency(A) for A in adj]
    if sequence_embeddings_path:
        train_embeddings_list_kpca = read_list(sequence_embeddings_path)
        features = replace_ft(train_embeddings_list_kpca, features)
    (
        features_train,
        features_valid,
        features_test,
        adj_train,
        adj_valid,
        adj_test,
        y_train,
        y_valid,
        proteins_test,
    ) = split_data(graph_labels_path, adj, features)

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(np.vstack(features_train))
    features_test_scaled = scaler.transform(np.vstack(features_test))
    features_valid_scaled = scaler.transform(np.vstack(features_valid))

    features_train_scaled = resize_prot_seq(features_train_scaled, features_train)
    features_test_scaled = resize_prot_seq(features_test_scaled, features_test)
    features_valid_scaled = resize_prot_seq(features_valid_scaled, features_valid)

    # Hyperparameters
    epochs = 100
    batch_size = 64
    n_hidden = 512
    n_input = 96
    dropout = 0.4
    learning_rate = 0.001
    n_class = 18
    weight_decay = 5e-3
    # Compute number of training and test samples
    N_train = len(adj_train)
    N_test = len(adj_test)
    N_valid = len(adj_valid)

    from sklearn.utils.class_weight import compute_class_weight

    sample_weights = (
        torch.tensor(
            compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        )
        .to(device)
        .float()
    )

    model = GNN(n_input, n_hidden, dropout, n_class, "GAT").to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.CrossEntropyLoss(weight=sample_weights)
    train(
        model,
        adj_train,
        adj_valid,
        features_train_scaled,
        features_valid_scaled,
        y_train,
        y_valid,
        optimizer,
        loss_function,
        epochs,
        N_train,
        N_valid,
        batch_size,
    )
    y_pred_proba = predict(model, adj_test, features_test_scaled, N_test, batch_size)
    write_sub(y_pred_proba, proteins_test, "sample_submissions_struct.csv")


if __name__ == "__main__":
    main()
