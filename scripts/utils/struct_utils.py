import scipy.sparse as sp
import numpy as np
import torch


def normalize_adjacency(A):
    """
    Function that normalizes an adjacency matrix
    """
    n = A.shape[0]
    A += sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D = sp.diags(inv_degs)
    A_normalized = D.dot(A)

    return A_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Function that converts a Scipy sparse matrix to a sparse Torch tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def replace_ft(list_sequences_emb, features):
    """
    Utils function to replace the one-hot-encoding by a custom encodings

    Arguments
    ---------
        list_sequences_emb: list(<np.array>)
            New embeddings
        features: np.array
            Current matrix of features

    Returns
    -------
        l_ft: list
            Updated features
    """
    l_ft = []
    for i, ft in enumerate(features):
        ft_del = np.delete(ft, obj=range(3, 23), axis=1)  # one hot encoding
        if list_sequences_emb[i][1:-1].shape[1] == 30:  # 13 sequences less than 30
            ft_emb = np.concatenate((ft_del, list_sequences_emb[i][1:-1]), axis=1)
            l_ft.append(ft_emb)
        else:
            ft = np.concatenate((ft, np.zeros((ft.shape[0], 10))), axis=1)
            l_ft.append(ft)
    return l_ft


def split_data(graph_labels_path, adj, features):
    """
    Custom train test split
    """
    adj_train = list()
    features_train = list()
    y_train = list()
    adj_test = list()
    features_test = list()
    proteins_test = list()
    with open(graph_labels_path, "r") as f:
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
    sep = int(len(features_train) * 0.25)
    features_train = features_train[:-sep]
    features_valid = features_train[-sep:]
    adj_train = adj_train[:-sep]
    adj_valid = adj_train[-sep:]
    y_train = y_train[:-sep]
    y_valid = y_train[-sep:]
    return (
        features_train,
        features_valid,
        features_test,
        adj_train,
        adj_valid,
        adj_test,
        y_train,
        y_valid,
        proteins_test,
    )


def resize_prot_seq(features_scaled, features):
    """
    Resize a flatten scaled features matrix based on the orignal shape of the matrix

    Arguments
    ---------
        features_scaled: list
            List of scaled vectors
        features: np.array
            Orginal numpy Array of features
    """
    res = []
    length_seq = [x.shape[0] for x in features]
    for size in length_seq:
        res.append(features_scaled[:size])
        features_scaled = features_scaled[size:]
    return res
