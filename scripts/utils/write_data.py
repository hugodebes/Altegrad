import csv
import pickle


def write_list(l, name_file):
    """
    Utils function to write a list onto a file

    Arguments
    ----------
        l : list
            List to write on the file
        name_file : str
            Name of the file
    """
    with open(name_file, "wb") as fp:
        pickle.dump(l, fp)
        print("Done writing list into a binary file")


def write_sub(y_pred_proba, proteins_test, file_path):
    """
    Write predictions to a file

    Arguments
    ---------
        y_pred_proba: np.array
            predicted class for the proteins in the test set
        proteins_test: list
            Name of the proteins in the test set
        file_path: str
            Path of the output file
    """
    with open(file_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        lst = list()
        for i in range(18):
            lst.append("class" + str(i))
        lst.insert(0, "name")
        writer.writerow(lst)
        for i, protein in enumerate(proteins_test):
            lst = y_pred_proba[i, :].tolist()
            lst.insert(0, protein)
            writer.writerow(lst)
