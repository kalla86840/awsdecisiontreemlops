import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def run(model_dir: str, test_path: str, output_dir: str):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    test_df = pd.read_csv(test_path)

    y_true, X_test = test_df["target"], test_df.drop(columns=["target"])
    preds = clf.predict(X_test)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    cm = confusion_matrix(y_true, preds).tolist()

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump({"metrics": {"accuracy": acc, "f1_macro": f1}, "confusion_matrix": cm}, f)

    print("Wrote evaluation.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    ap.add_argument("--test-path", type=str, default="/opt/ml/processing/test/test.csv")
    ap.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    a = ap.parse_args()
    run(a.model_dir, a.test_path, a.output_dir)
