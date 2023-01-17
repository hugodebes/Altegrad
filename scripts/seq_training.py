import csv
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from seq_read import read_data_sequences, read_data_structures
from seq_mapping import mapping

sequences_train, sequences_test, proteins_test, y_train, idx_train = read_data_sequences()
number_of_nodes, number_of_edges_per_protein, edge_distance_attr_per_protein = read_data_structures()

X_train, X_test = mapping(sequences_train, sequences_test)

def embedding(X_train, X_test, threshold):

    pca = PCA().fit(X_train.toarray())
    X_train_e = pca.transform(X_train.toarray())
    X_test_e = pca.transform(X_test.toarray())

    variance_ratio = pca.explained_variance_ratio_.cumsum()
    n = len(variance_ratio[variance_ratio < threshold])

    return X_train_e[:,:n], X_test_e[:,:n]

#X_train_embedded, X_test_embedded = embedding(X_train, X_test, 0.9)

new_features_train = np.concatenate(
    (number_of_nodes[idx_train].reshape(-1,1),
     number_of_edges_per_protein[idx_train].reshape(-1,1),
     edge_distance_attr_per_protein[idx_train].reshape(-1,1)), axis = 1)

new_features_test = np.concatenate(
    (np.delete(number_of_nodes, idx_train).reshape(-1,1),
     np.delete(number_of_edges_per_protein, idx_train).reshape(-1,1),
     np.delete(edge_distance_attr_per_protein, idx_train).reshape(-1,1)), axis = 1)

scaler = RobustScaler()
new_features_train = scaler.fit_transform(new_features_train)
new_features_test = scaler.transform(new_features_test)

X_train_nf = np.zeros((X_train.shape[0], X_train.shape[1] + 3))
X_test_nf = np.zeros((X_test.shape[0], X_test.shape[1] + 3))

X_train_nf[:, :X_train.shape[1]] = X_train
X_test_nf[:, :X_test.shape[1]] = X_test

X_train_nf[:, -3:] = new_features_train
X_test_nf[:, -3:] = new_features_test

svc = SVC(probability=True).fit(X_train_nf, y_train)
y_pred_proba = svc.predict_proba(X_test_nf)

# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = list()
    for i in range(18):
        lst.append('class'+str(i))
    lst.insert(0, "name")
    writer.writerow(lst)
    for i, protein in enumerate(proteins_test):
        lst = y_pred_proba[i,:].tolist()
        lst.insert(0, protein)
        writer.writerow(lst)

