from sklearn import ensemble

MODELS = {
    "randomForest" : ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extraTree" : ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}