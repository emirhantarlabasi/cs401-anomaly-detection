from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid


def grid_search_isolation_forest(
    train_path: Path = Path("data/train_optimized.csv"),
    test_path: Path = Path("data/test_optimized.csv"),
    output_path: Path = Path("reports/tables/if_grid_search_results.csv"),
) -> dict:
    print("Grid search basliyor...")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Optimize edilmis train/test dosyalari bulunamadi.")

    X_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y_test = df_test["Attack Type"].apply(lambda x: 0 if x == "Normal Traffic" else 1)
    X_test = df_test.drop(columns=["Attack Type"], errors="ignore")
    X_train = X_train.drop(columns=["Attack Type"], errors="ignore")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "contamination": [0.01, 0.05, 0.10, 0.15],
        "max_samples": [256, 512, "auto"],
    }

    records: list[dict] = []
    best_score = -1.0
    best_params: dict = {}

    all_params = list(ParameterGrid(param_grid))
    print(f"Toplam kombinasyon: {len(all_params)}")

    for params in all_params:
        print(f"Testing: {params} ...", end=" ")
        clf = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            max_samples=params["max_samples"],
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train)
        y_pred_raw = clf.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred_raw]
        score = f1_score(y_test, y_pred, pos_label=1)
        print(f"F1: {score:.4f}")

        record = dict(params)
        record["f1_attack"] = score
        records.append(record)

        if score > best_score:
            best_score = score
            best_params = params

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values("f1_attack", ascending=False).to_csv(output_path, index=False)

    print("=" * 40)
    print(f"En iyi F1 (attack): {best_score:.4f}")
    print(f"En iyi ayarlar: {best_params}")
    print(f"Grid search tablo cikisi: {output_path}")
    print("=" * 40)

    return {"best_score": best_score, "best_params": best_params}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for Isolation Forest")
    parser.add_argument("--train-path", type=Path, default=Path("data/train_optimized.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test_optimized.csv"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/tables/if_grid_search_results.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    grid_search_isolation_forest(
        train_path=args.train_path,
        test_path=args.test_path,
        output_path=args.output_path,
    )