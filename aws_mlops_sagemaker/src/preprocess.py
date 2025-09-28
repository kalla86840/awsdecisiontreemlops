import argparse
import os
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

def _load_iris_df():
    iris = datasets.load_iris(as_frame=True)
    X = iris.data
    y = iris.target.rename("target")
    return pd.concat([X, y], axis=1)

def _load_cars_df(cars_csv: str | None):
    cols = ["engine_disp_l", "horsepower", "weight_kg", "cylinders", "length_mm", "target"]
    if cars_csv and os.path.exists(cars_csv):
        df = pd.read_csv(cars_csv)
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"cars.csv missing columns: {missing}")
        return df[cols].copy()

    import numpy as np
    rng = np.random.default_rng(0)
    n = 200
    cyl = rng.choice([3, 4, 4, 4, 6, 6, 8], size=n)
    eng = np.round(rng.uniform(1.0, 5.5, size=n) + (cyl - 4) * 0.1, 2)
    hp = np.clip((150 + (eng - 2.0) * 35 + rng.normal(0, 25, size=n)).astype(int), 60, 550)
    wt = np.clip((1400 + (cyl - 4) * 120 + (eng - 2.0) * 110 + rng.normal(0, 120, size=n)).astype(int), 900, 2600)
    ln = np.clip((4300 + (cyl - 4) * 80 + rng.normal(0, 150, size=n)).astype(int), 3600, 5300)
    score = 0.004 * hp + 0.0015 * wt + 0.0008 * ln + 0.7 * (cyl >= 6) + 0.6 * (eng >= 3.0)
    target = (score > 6.2).astype(int)
    return pd.DataFrame({
        "engine_disp_l": eng, "horsepower": hp, "weight_kg": wt,
        "cylinders": cyl, "length_mm": ln, "target": target
    })

def run(output_base: str, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42,
        dataset: str = "iris", cars_csv: str | None = None):
    df = _load_cars_df(cars_csv) if dataset == "cars" else _load_iris_df()
    train_full, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["target"])
    train, val = train_test_split(train_full, test_size=val_size, random_state=random_state, stratify=train_full["target"])

    for name, d in [("train", train), ("validation", val), ("test", test)]:
        path = os.path.join(output_base, name, f"{name}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d.to_csv(path, index=False)
        print("Wrote", path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-base", type=str, default="/opt/ml/processing")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--dataset", type=str, choices=["iris", "cars"], default="iris")
    p.add_argument("--cars-csv", type=str, default=None)
    a = p.parse_args()
    run(a.output_base, a.test_size, a.val_size, a.random_state, a.dataset, a.cars_csv)
