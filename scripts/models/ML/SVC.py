from sklearn.svm import SVC

def fine_tune_svc():
    return SVC(C=1000, gamma=0.1, probability=True, kernel="rbf")
    