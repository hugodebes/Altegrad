import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from protlearn.features import *
from sklearn.preprocessing import RobustScaler

def mapping(sequences_train, sequences_test):

    vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    X_train = vec.fit_transform(sequences_train)
    X_test = vec.transform(sequences_test)

    comp, _ = aac(sequences_train, remove_zero_cols=True)
    aaind, _ = aaindex1(sequences_train, standardize='zscore')
    apaac_comp, _ = apaac(sequences_train, lambda_=8, remove_zero_cols=True)
    atoms, _ = atc(sequences_train)
    ctd_arr, _ = ctd(sequences_train)
    c, _ = ctdc(sequences_train)
    ent = entropy(sequences_train)
    gearyC = geary(sequences_train)
    moranI = moran(sequences_train)
    mb = moreau_broto(sequences_train)
    sw, _ = socn(sequences_train, d=8)

    comp_t, _ = aac(sequences_test, remove_zero_cols=True)
    aaind_t, _ = aaindex1(sequences_test, standardize='zscore')
    apaac_comp_t, _ = apaac(sequences_test, lambda_=8, remove_zero_cols=True)
    atoms_t, _ = atc(sequences_test)
    ctd_arr_t, _ = ctd(sequences_test)
    c_t, _ = ctdc(sequences_test)
    ent_t = entropy(sequences_test)
    gearyC_t = geary(sequences_test)
    moranI_t = moran(sequences_test)
    mb_t = moreau_broto(sequences_test)
    sw_t, _ = socn(sequences_test, d=8)

    features_train = [comp, aaind, apaac_comp, atoms, ctd_arr, c, ent, gearyC, moranI, mb, sw]
    features_test = [comp_t, aaind_t, apaac_comp_t, atoms_t, ctd_arr_t, c_t, ent_t, gearyC_t, moranI_t, mb_t, sw_t]

    n = 0
    for f in features_train:
        n+=f.shape[1]
    
    X_train_f = np.zeros((X_train.shape[0], X_train.shape[1] + n))
    X_test_f = np.zeros((X_test.shape[0], X_test.shape[1] + n))

    X_train_f[:, :X_train.shape[1]] = X_train.toarray()
    X_test_f[:, :X_test.shape[1]] = X_test.toarray()

    tmp = n
    for f, g in zip(features_train, features_test):
        if -tmp+f.shape[1] == 0:
            X_train_f[:, -tmp:] = f
            X_test_f[:, -tmp:] = g
        else:
            X_train_f[:, -tmp:-tmp+f.shape[1]] = f
            X_test_f[:, -tmp:-tmp+g.shape[1]] = g
        tmp -= f.shape[1]


    scaler = RobustScaler()
    X_train_f = scaler.fit_transform(X_train_f)
    X_test_f = scaler.transform(X_test_f)
    
    return X_train_f, X_test_f
