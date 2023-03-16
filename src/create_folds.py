import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":

    df = pd.read_csv("input/train.csv")
    df['kfold'] = -1 
    # shuffle the data and drop index
    df = df.sample(frac=1).reset_index(drop = True)
    # create splits
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # create folda and add fold attributes to fold column.
    for fold , (X_trn_idx, y_trn_idx) in enumerate (kf.split(X = df, y = df.target.values)):
        print("Fold #:",fold)
        print(len(X_trn_idx), len(y_trn_idx))
        df.loc[y_trn_idx,"kfold"] = fold
    # save the folds in a csv file
    df.to_csv("input/train_folds.csv", index=False)