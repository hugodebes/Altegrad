import numpy as np
import scipy.sparse as sp
import pickle


def read_data_sequences():
    # Read sequences
    sequences = list()
    with open("data/sequences.txt", "r") as f:
        for line in f:
            sequences.append(line[:-1])

    # Split data into training and test sets
    sequences_train = list()
    sequences_test = list()
    proteins_test = list()
    y_train = list()
    with open("data/graph_labels.txt", "r") as f:
        for i, line in enumerate(f):
            t = line.split(",")
            if len(t[1][:-1]) == 0:
                proteins_test.append(t[0])
                sequences_test.append(sequences[i])
            else:
                sequences_train.append(sequences[i])
                y_train.append(int(t[1][:-1]))
    return sequences_train, sequences_test, proteins_test, y_train


def read_data_structures():
    """
    Function that loads graphs
    """
    graph_indicator = np.loadtxt("graph_indicator.txt", dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt("edgelist.txt", dtype=np.int64, delimiter=",")
    edges_inv = np.vstack((edges[:, 1], edges[:, 0]))
    edges = np.vstack((edges, edges_inv.T))
    s = edges[:, 0] * graph_indicator.size + edges[:, 1]
    idx_sort = np.argsort(s)
    edges = edges[idx_sort, :]
    edges, idx_unique = np.unique(edges, axis=0, return_index=True)
    A = sp.csr_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(graph_indicator.size, graph_indicator.size),
    )

    x = np.loadtxt("node_attributes.txt", delimiter=",")
    edge_attr = np.loadtxt("edge_attributes.txt", delimiter=",")
    edge_attr = np.vstack((edge_attr, edge_attr))
    edge_attr = edge_attr[idx_sort, :]
    edge_attr = edge_attr[idx_unique, :]

    adj = []
    features = []
    edge_features = []
    idx_n = 0
    idx_m = 0
    for i in range(graph_size.size):
        adj.append(A[idx_n : idx_n + graph_size[i], idx_n : idx_n + graph_size[i]])
        edge_features.append(edge_attr[idx_m : idx_m + adj[i].nnz, :])
        features.append(x[idx_n : idx_n + graph_size[i], :])
        idx_n += graph_size[i]
        idx_m += adj[i].nnz

    return adj, features, edge_features


# Read list to memory
def read_list(file_name):
    # for reading also binary mode is important
    with open(file_name, "rb") as fp:
        n_list = pickle.load(fp)
        return n_list


def read_data_without_unnatural():
    # Read sequences
    sequences = list()
    with open("data/sequences.txt", "r") as f:
        for line in f:
            sequences.append(line[:-1])

    # Split data into training and test sets
    sequences_train = list()
    sequences_test = list()
    proteins_test = list()
    y_train = list()
    idx_train = list()
    with open("data/graph_labels.txt", "r") as f:
        for i, line in enumerate(f):
            t = line.split(",")
            if len(t[1][:-1]) == 0:
                proteins_test.append(t[0])
                sequences_test.append(sequences[i])
            else:
                idx_train.append(i)
                sequences_train.append(sequences[i])
                y_train.append(int(t[1][:-1]))

    #Remove unnatural amino acid
    for i, sequences in enumerate(sequences_train):
        if 'X' in sequences:
            sequences_train[i] = sequences.replace("X","") 

    for i, sequences in enumerate(sequences_test):
        if 'X' in sequences:
            sequences_test[i] = sequences.replace("X","")    

    return sequences_train, sequences_test, proteins_test, y_train, idx_train
