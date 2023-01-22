# basics
import csv
import numpy as np

# sklearn
from sklearn.preprocessing import RobustScaler

# methods
from utils.read_data import read_data_without_unnatural
from utils.struct_utils import new_structure_features
from models.ML.SVC import fine_tune_svc
from feature_extraction.seq_mapping  import mapping, embedding
from utils.write_data import write_sub

def main():

    # Getting the new features with the structure
    sequences_train, sequences_test, proteins_test, y_train, idx_train = read_data_without_unnatural()
    number_of_nodes, number_of_edges_per_protein, edge_distance_attr_per_protein = new_structure_features()

    # New features with protlearn
    X_train, X_test = mapping(sequences_train, sequences_test)

    # Embedding
    X_train_embedded, X_test_embedded = embedding(X_train, X_test, 0.9)


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

    svc = fine_tune_svc().fit(X_train_nf, y_train)
    y_pred_proba = svc.predict_proba(X_test_nf)
    write_sub(y_pred_proba, proteins_test, "sample_submissions_struct.csv")

    
if __name__ == "__main__":
    main()