from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sequences_train, sequences_test):
    # Map sequences to
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    X_train = vec.fit_transform(sequences_train)
    X_test = vec.transform(sequences_test)
    return X_train, X_test
