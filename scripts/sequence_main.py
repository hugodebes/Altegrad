# basics
import numpy as np


# Torch
import torch
import torch.nn as nn
import torch.optim as optim

from models.deep_learning.HAN import HAN
from models.deep_learning.LSTM import LSTMClassifier
from utils.read_data import read_data_sequences
from sequence_train_eval import train

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model_type = "LSTM"  
model_type = "HAN"  


class Dataset_(Dataset):
    def __init__(self, x, y):
        self.proteins = x
        self.labels = y

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        protein = self.proteins[index]
        label = self.labels[index] 
        sample = {
            "protein": torch.tensor(protein),
            "label": torch.tensor(label).type(torch.LongTensor),
            }
        return sample


def get_loader(x, y, batch_size=32):
    dataset = Dataset_(x, y)
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            )
    return data_loader


def main():

    #Read the data
    sequences_train, sequences_test, proteins_test, y_train = read_data_sequences()

    if model_type == "LSTM": 

        #create the vocabulary
        aa_to_id = {k: i+1 for i, k in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
        id_to_aa = {v: k for k, v in aa_to_id.items()}

        #adapt the data to the vocabulary
        sequences_train = [[aa_to_id[aa] for aa in prot] for prot in sequences_train]
        sequences_test = [[aa_to_id[aa] for aa in prot] for prot in sequences_test]

        #Now we can add the padding
        length_padding = max([len(prot)for prot in sequences_train])
        sequences_train = np.array([np.pad(prot, (0, length_padding - len(prot)), constant_values=(0,0)) for prot in sequences_train])
        sequences_test = np.array([np.pad(prot, (0, length_padding - len(prot)), constant_values=(0,0)) for prot in sequences_test])

        #let's create a train and a valid dataset
        X_train, X_valid, y_train, y_valid = train_test_split(sequences_train, torch.Tensor(y_train), test_size=0.2)
        input_size = X_train.shape

        # Hyperparameters
        batch_size = 64
        hidden_dim = 30
        lstm_layers = 2
        max_epochs = 15

        training_set = Dataset_(X_train, y_train)
        test_set = Dataset_(X_valid, y_valid)
        loader_training = DataLoader(training_set, batch_size=batch_size)
        loader_test = DataLoader(test_set)

        model = LSTMClassifier(batch_size, hidden_dim, lstm_layers, length_padding)
        lr = 0.001  # learning rate
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train(model, model_type, loader_training, loader_test, optimizer, criterion, max_epochs)

    if model_type == "HAN": 

        #create the vocabulary
        aa_to_id = {k: i+1 for i, k in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
        id_to_aa = {v: k for k, v in aa_to_id.items()}
        id_to_aa[0] = "<PAD>"

        #adapt the data to the vocabulary
        sequences_train = [[aa_to_id[aa] for aa in prot] for prot in sequences_train]
        sequences_test = [[aa_to_id[aa] for aa in prot] for prot in sequences_test]

        #separate in three parts
        sequences_train = np.array([[prot[i*(len(prot)//3) : (i+1)*(len(prot)//3)]for i in range(3)] for prot in sequences_train], dtype='object')
        sequences_test = np.array([[prot[i*(len(prot)//3) : (i+1)*(len(prot)//3)]for i in range(3)] for prot in sequences_test], dtype='object')

        #Now we can add the padding
        length_padding = 330
        sequences_train = np.array([[np.pad(part, (0, length_padding - len(part)), constant_values=(0,0)) for part in prot] for prot in sequences_train])
        sequences_test= np.array([[np.pad(part, (0, length_padding - len(part)), constant_values=(0,0)) for part in prot] for prot in sequences_test])

        #let's create a train and a valid dataset
        X_train, X_valid, y_train, y_valid = train_test_split(sequences_train, torch.Tensor(y_train), test_size=0.2)
        input_size = X_train.shape

        # Hyperparameters
        d = 100 # dimensionality of word embeddings
        n_units = 30 # RNN layer dimensionality
        drop_rate = 0.5 # dropout
        padding_idx = 0
        batch_size = 64
        max_epochs = 50

        training_set = Dataset_(X_train, y_train)
        test_set = Dataset_(X_valid, y_valid)
        loader_training = DataLoader(training_set, batch_size=batch_size)
        loader_test = DataLoader(test_set)

        model = HAN(input_size, n_units, id_to_aa, d, drop_rate)
        lr = 0.001  # learning rate
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train(model, model_type, loader_training, loader_test, optimizer, criterion, max_epochs)
    

if __name__ == "__main__":
    main()
