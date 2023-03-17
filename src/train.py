import os 
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3],
}

if __name__ == "__main__":
    # get the traning data from environment variables
    df = pd.read_csv(TRAINING_DATA)
    trn_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    tst_df = df[df.kfold == FOLD]
    
    y_trn = trn_df.target.values
    y_tst = tst_df.target.values

    tst_df = tst_df.drop(["id","target","kfold"],axis=1)
    trn_df = trn_df.drop(["id","target","kfold"],axis=1)

    label_encoders = []
    for col in trn_df.columns:
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(trn_df[col].values.tolist() + tst_df[col].values.tolist())
        trn_df.loc[:, col] = lbl_enc.transform(trn_df[col].values.tolist())
        tst_df.loc[:, col] = lbl_enc.transform(tst_df[col].values.tolist())
        label_encoders.append((col,lbl_enc))

    # train data
    # by increasing the number of estimator would increase the auccuracy score
    clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2)
    clf.fit(trn_df,y_trn)
    y_preds = clf.predict_proba(tst_df)[:,1]
    print(metrics.roc_auc_score(y_tst, y_preds))


    