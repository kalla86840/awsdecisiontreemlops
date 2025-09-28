import argparse
import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def run(train_path, val_path, model_dir, max_depth=0, min_samples_leaf=1, random_state=42):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    y_train, X_train = train_df["target"], train_df.drop(columns=["target"])
    y_val, X_val = val_df["target"], val_df.drop(columns=["target"])

    depth = None if (max_depth is None or int(max_depth) <= 0) else int(max_depth)
    clf = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_val, clf.predict(X_val))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        f.write(f'{{"validation_accuracy": {acc:.4f}}}')

    print(f"Saved model; val_acc={acc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", type=str, default="/opt/ml/input/data/train/train.csv")
    ap.add_argument("--val-path", type=str, default="/opt/ml/input/data/validation/validation.csv")
    ap.add_argument("--model-dir", type=str, default="/opt/ml/model")
    ap.add_argument("--max-depth", type=int, default=0, help="<=0 means None")
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--random-state", type=int, default=42)
    a = ap.parse_args()
    run(a.train_path, a.val_path, a.model_dir, a.max_depth, a.min_samples_leaf, a.random_state)
