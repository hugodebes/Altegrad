import time
import torch
import scipy.sparse as sp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_accuracy(data_loader, model, model_type, verbose=True):
  
    model.eval()
    accuracies = []

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):

            y = []
            if model_type == "LSTM": 
                y = model(data['protein'])
            elif model_type == "HAN":
                y = model(data['protein'])[0] 
                y = y[:, -1] # only last vector
            else: 
                raise ValueError("The model type doesn't exist")

            accuracy = torch.sum(torch.argmax(y,axis=1) == data['label']) * 100 / len(data['label'])
            accuracies.append(accuracy)

        return np.mean(accuracies)


def train(model, model_type, data_loader_train, data_loader_test, optimizer, criterion, max_epochs):

    for epoch in range(max_epochs):

        model.train()
        losses = []
        accuracies = []

        for batch_idx, data in enumerate(data_loader_train):

            x = data['protein'].type(torch.LongTensor)
            y = data['label'].type(torch.LongTensor)

            y_pred = []
            if model_type == "LSTM": 
                y_pred = model(x)
            elif model_type == "HAN":
                y_pred = model.forward(x)[0]
                y_pred = y_pred[:, -1]
            else: 
                raise ValueError("The model type doesn't exist")


            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            accuracy = torch.sum(torch.argmax(y_pred,axis=1) == y) * 100 / len(y)
            losses.append(loss.item())
            accuracies.append(accuracy)

            if (batch_idx % 16 == 0):
                print("===> Epoch {} /{}/: Avg. Loss: {:.4f}, Avg. Accuracy: {:.4f}%"
                    .format(epoch, batch_idx, np.mean(losses), np.mean(accuracies)))

        test_acc = evaluate_accuracy(data_loader_test, model, model_type, False)
        print("==============================================================================")
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. Accuracy: {:.4f}, Validation Accuracy: {:3.2f}%"
            .format(epoch, np.mean(losses), np.mean(accuracies), test_acc))
        print("==============================================================================")



