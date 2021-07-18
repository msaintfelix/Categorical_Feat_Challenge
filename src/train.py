import os
import joblib
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")


if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)

    train_df = df[df['val_kfold']!=FOLD]
    val_df = df[df['val_kfold']==FOLD]

    ytrain = train_df.target.values
    yval = val_df.target.values

    train_df = train_df.drop(["id", "target", "val_kfold"], axis=1)
    val_df = val_df.drop(["id", "target", "val_kfold"], axis=1)

    val_df = val_df[train_df.columns]

    # initialize a dictionary for later access with str
    label_encoders = {}

    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()

        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        val_df.loc[:, c] = val_df.loc[:, c].astype(str).fillna("NONE")
        test_df.loc[:, c] = test_df.loc[:, c].astype(str).fillna("NONE")

        # the fit() method takes lists as input. Let's concatenate train, val and test lists to fit all possible values.
        lbl.fit(train_df[c].values.tolist() + val_df[c].values.tolist() + test_df[c].values.tolist())

        # let's transform each column accordingly
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        val_df.loc[:, c] = lbl.transform(val_df[c].values.tolist())

        # store the original columns and associated labels
        # as per sklearn doc, each col content can be accessed via lbl.classes_
        label_encoders[c] = lbl

    clf = dispatcher.MODEL_DICT[MODEL]
    clf.fit(train_df, ytrain)

    # display probabilities for class 1 target.
    predictions = clf.predict_proba(val_df)[:, 1]
    print(metrics.roc_auc_score(yval, predictions))

    # serialize joblib.dump(object, path)
    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")