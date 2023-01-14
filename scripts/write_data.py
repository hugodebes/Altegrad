import csv


def write_sub(y_pred_proba, proteins_test):
    # Write predictions to a file
    with open("sample_submission.csv", "w") as csvfile:
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
