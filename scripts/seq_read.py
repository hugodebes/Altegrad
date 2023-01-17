import numpy as np

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

def read_data_structures():
    """
    Function that loads graphs
    """
    graph_indicator = np.loadtxt("data/graph_indicator.txt", dtype=np.int64)
    _, number_of_nodes = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt("data/edgelist.txt", dtype=np.int64, delimiter=",")
    _, number_of_edges = np.unique(edges[:,0], return_counts=True)

    edge_attributes = np.loadtxt("data/edge_attributes.txt", dtype=np.float64, delimiter=",")
    edge_distance_attr = edge_attributes[:,0]

    number_of_edges_per_protein = list()
    start = 0
    for elt in number_of_nodes:
        number_of_edges_per_protein.append(number_of_edges[start:start+elt].sum())
        start += elt
    
    edge_distance_attr_per_protein = list()
    start = 0
    for elt in number_of_edges_per_protein:
        edge_distance_attr_per_protein.append(np.median(edge_distance_attr[start:start+elt]))
        start += elt

    return number_of_nodes, np.array(number_of_edges_per_protein), np.array(edge_distance_attr_per_protein)


