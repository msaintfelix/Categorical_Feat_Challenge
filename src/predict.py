import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.environ.get("MODEL_PATH")
TEST_DATA = os.environ.get("TEST_DATA")


def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    predictions = None
    test_idx = df["id"].values

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(test_data_path)
        encoders = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))

        for c in encoders:
            print(c)
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))

        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub


if __name__ == "__main__":
    submission = predict(test_data_path=TEST_DATA,
                         model_type="randomforest",
                         model_path=MODEL_PATH)

    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)

    submission.to_csv(f"models/rf_submission.csv", index=False)
