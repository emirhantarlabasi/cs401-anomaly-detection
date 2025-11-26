"""
Baseline anomaly detection pipeline built around Isolation Forest.

This module mirrors the exploratory notebook steps so the same workflow can
be reproduced from the command line or imported inside other scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


def load_data(path: Path, row_limit: int | None = None) -> pd.DataFrame:
    """Read the CICIDS dataset with an optional row cap."""
    read_kwargs = {"low_memory": False}
    if row_limit is not None:
        read_kwargs["nrows"] = row_limit
    df = pd.read_csv(path, **read_kwargs)
    return df


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a 0/1 label derived from Attack Type."""
    df = df.copy()
    df["label"] = df["Attack Type"].apply(lambda x: 0 if x == "Normal Traffic" else 1)
    return df


def prepare_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, list]:
    """Split into train/test matrices using only feature columns."""
    feature_cols = df.columns.drop(["Attack Type", "label"])
    train_df = df[df["label"] == 0]
    test_df = df.copy()
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]
    return X_train, X_test, y_test, feature_cols.tolist()


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.36,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    """Fit Isolation Forest on normal traffic only."""
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    X_train_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    iso.fit(X_train_np)
    return iso


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[np.ndarray, str]:
    """Compute confusion matrix and classification report."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3)
    return cm, report


def run_baseline(
    data_path: Path,
    row_limit: int | None = None,
    contamination: float = 0.36,
    report_path: Optional[Path] = None,
) -> None:
    """Execute the full pipeline and print evaluation outputs."""
    print(f"Loading data from {data_path} (row_limit={row_limit})")
    df = load_data(data_path, row_limit=row_limit)
    df = add_binary_label(df)
    label_dist = df["label"].value_counts(normalize=True).mul(100).round(2)
    print("Label distribution (%):")
    print(label_dist.to_string())

    X_train, X_test, y_test, feature_cols = prepare_train_test(df)
    print(f"Using {len(feature_cols)} features -> train: {X_train.shape}, test: {X_test.shape}")

    iso = train_isolation_forest(X_train, contamination=contamination)
    raw_pred = iso.predict(X_test.to_numpy())
    y_pred = np.where(raw_pred == 1, 0, 1)

    cm, report = evaluate(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    if report_path:
        report_content = [
            "Label distribution (%)",
            label_dist.to_string(),
            "",
            "Confusion matrix:",
            np.array2string(cm),
            "",
            "Classification report:",
            report,
        ]
        report_path.write_text("\n".join(report_content), encoding="utf-8")
        print(f"\nSaved baseline report to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline Isolation Forest anomaly detector")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../data/cicids2017_cleaned.csv"),
        help="Path to cleaned CICIDS2017 CSV file",
    )
    parser.add_argument("--row-limit", type=int, default=None, help="Optionally cap number of rows to load")
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.36,
        help="Expected anomaly ratio passed to Isolation Forest",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("baseline_report.txt"),
        help="Destination file to store confusion matrix and classification report",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(
        data_path=args.data_path,
        row_limit=args.row_limit,
        contamination=args.contamination,
        report_path=args.report_path,
    )

