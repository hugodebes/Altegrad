import time
import torch
import scipy.sparse as sp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_accuracy(data_loader, model, verbose=True):
  
    model.eval()
    accuracies = []

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader_train):

        	x = data['protein'].type(torch.LongTensor)
        	y = data['label'].type(torch.LongTensor)

        	y_pred = model(x_batch)
            accuracy = torch.sum(torch.argmax(y_pred, axis=1) == y) * 100 / len(y)
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
        	if model_type == "HAN":
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
            	print("===> Epoch {} {}/{}: Avg. Loss: {:.4f}, Avg. Accuracy: {:.4f}%"
                	.format(epoch, batch_idx, batch_size, np.mean(losses), np.mean(accuracies)))

    	test_acc = evaluate_accuracy(loader_test, model, False)
    	print("==============================================================================")
    	print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. Accuracy: {:.4f}, Validation Accuracy: {:3.2f}%"
              .format(epoch, np.mean(losses), np.mean(accuracies), test_acc))
    	print("==============================================================================")



def eval_model(model, adj, features, y, N, batch_size, loss_function):
    model.eval()
    correct = 0
    count = 0
    total_epoch_loss = 0
    with torch.no_grad():
        # Iterate over the batches
        for i in range(0, N, batch_size):
            adj_batch = list()
            features_batch = list()
            idx_batch = list()
            y_batch = list()

            # Create tensors
            for j in range(i, min(N, i + batch_size)):
                n = adj[j].shape[0]
                adj_batch.append(adj[j] + sp.identity(n))
                features_batch.append(features[j])
                idx_batch.extend([j - i] * n)
                y_batch.append(y[j])

            adj_batch = sp.block_diag(adj_batch)
            features_batch = np.vstack(features_batch)

            adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device)
            features_batch = torch.FloatTensor(features_batch).to(device)
            idx_batch = torch.LongTensor(idx_batch).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            output = model(features_batch, adj_batch, idx_batch)
            loss = loss_function(output, y_batch)
            total_epoch_loss += loss.item() * output.size(0)
            count += output.size(0)
            preds = output.max(1)[1].type_as(y_batch)
            correct += torch.sum(preds.eq(y_batch).double())
    return total_epoch_loss / (i + 1), 100 * correct / count


def predict(model, adj_test, features_test_scaled, N_test, batch_size):
    print("Evaluate model")
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
            features_batch.append(features_test_scaled[j])
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
    return y_pred_proba
