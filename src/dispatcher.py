from sklearn import ensemble

MODEL_DICT = {
    "randomforest" : ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees" : ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2)
}