import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from read_data import read_data_sequences
from write_data import write_sub
from seq_dataset import ProteinSeqdataset
from seq_model import ProteinLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, opt, criterion, dataloader):
    model.train()
    losses = []
    for i, (x, y, len_) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        len_ = len_.to(device)
        opt.zero_grad()
        # (1) Forward
        pred = model.forward(x, len_)
        # (2) Compute the loss
        loss = criterion(pred, y)
        # (3) Compute gradients with the criterion
        loss.backward()
        # (4) Update weights with the optimizer
        opt.step()
        losses.append(loss.item())
        # Count the number of correct predictions in the batch - here, you'll need to use the sigmoid
        num_corrects = (torch.round(torch.sigmoid(pred)) == y).float().sum()
        acc = 100.0 * num_corrects / len(y)

        if i % 20 == 0:
            print(
                "Batch "
                + str(i)
                + " : training loss = "
                + str(loss.item())
                + "; training acc = "
                + str(acc.item())
            )
    return losses


# Same for the evaluation ! We don't need the optimizer here.
def eval_model(model, criterion, evalloader):
    model.eval()
    total_epoch_loss = 0
    total_epoch_acc = 0
    with torch.no_grad():
        for i, (x, y, len_) in enumerate(evalloader):
            x, y = x.to(device), y.to(device)
            len_ = len_.to(device)
            pred = model.forward(x, len_)
            loss = criterion(pred, y)
            num_corrects = (torch.round(torch.sigmoid(pred)) == y).float().sum()
            acc = 100.0 * num_corrects / len(y)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / (i + 1), total_epoch_acc / (i + 1)


# A function which will help you execute experiments rapidly - with a early_stopping option when necessary.
def experiment(
    model,
    training_dataloader,
    valid_dataloader,
    opt,
    criterion,
    num_epochs=5,
    early_stopping=True,
):
    train_losses = []
    if early_stopping:
        best_valid_loss = 10.0
    print("Beginning training...")
    for e in range(num_epochs):
        print("Epoch " + str(e + 1) + ":")
        train_losses += train_epoch(model, opt, criterion, training_dataloader)
        valid_loss, valid_acc = eval_model(model, criterion, valid_dataloader)
        print(
            "Epoch "
            + str(e + 1)
            + " : Validation loss = "
            + str(valid_loss)
            + "; Validation acc = "
            + str(valid_acc)
        )
        if early_stopping:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
            else:
                print("Early stopping.")
                break
    return train_losses


def main():
    sequences_train, sequences_test, proteins_test, y_train = read_data_sequences()

    train_x, val_x, train_y, val_y = train_test_split(
        np.array(sequences_train), np.array(y_train), test_size=0.1
    )
    print("Training:", train_x.shape, train_y.shape)
    print("Validation:", val_x.shape, val_y.shape)

    train_dataset = ProteinSeqdataset(train_x, train_y)
    val_dataset = ProteinSeqdataset(val_x, val_y)
    aa2id, id2aa = train_dataset.get_vocab()
    nb_class = len(set(y_train))
    vocab_size = len(aa2id)

    training_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=25)

    # Instantiate the model
    model = ProteinLSTM(
        hidden_size=64,
        num_layers=2,
        num_classes=nb_class,
        vocab_size=vocab_size,
        embedding_dim=64,
    )
    # Create an optimizer
    opt = optim.Adam(model.parameters(), lr=0.0025, betas=(0.9, 0.999))
    # The criterion is a binary cross entropy loss based on logits - meaning that the sigmoid is integrated into the criterion
    criterion = nn.BCEWithLogitsLoss()

    train_losses = experiment(
        model, training_dataloader, valid_dataloader, opt, criterion, num_epochs=10
    )
    print(train_losses)

    write_sub(y_pred_proba, proteins_test)


if __name__ == "__main__":

    main()
