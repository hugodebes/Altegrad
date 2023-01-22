import time
import torch
import scipy.sparse as sp
import numpy as np

from scripts.utils.struct_utils import sparse_mx_to_torch_sparse_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
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
):
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
                features_batch.append(features_train_scaled[j])
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
                "acc_train: {:.4f}".format(100 * correct / count),
                "time: {:.4f}s".format(time.time() - t),
            )
            if epoch != 0:
                total_epoch_loss, total_epoch_acc = eval_model(
                    model,
                    adj_valid,
                    features_valid_scaled,
                    y_valid,
                    N_valid,
                    batch_size,
                    loss_function,
                )
                print(
                    "Validation:",
                    "loss_valid: {:.4f}".format(total_epoch_loss),
                    "acc_valid: {:.4f}".format(total_epoch_acc),
                )


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
