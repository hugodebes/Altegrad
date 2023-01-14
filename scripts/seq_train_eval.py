from sklearn.linear_model import LogisticRegression

from read_data import read_data_sequences
from seq_preprocess import tf_idf
from write_data import write_sub

sequences_train, sequences_test, proteins_test, y_train = read_data_sequences()

X_train, X_test = tf_idf(sequences_train, sequences_test)

# Train a logistic regression classifier and use the classifier to
# make predictions
clf = LogisticRegression(solver="liblinear")
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)

write_sub(y_pred_proba, proteins_test)
