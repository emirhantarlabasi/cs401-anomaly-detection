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

from data_preprocessing import (
    add_binary_label as _add_binary_label,
    load_data as _load_data,
    prepare_train_test as _prepare_train_test,
)
from evaluate import compute_metrics_dict, evaluate_binary, save_baseline_report


def load_data(path: Path, row_limit: int | None = None) -> pd.DataFrame:
    """Backward-compatible wrapper around shared preprocessing utility."""
    return _load_data(path, row_limit=row_limit)


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper around shared preprocessing utility."""
    return _add_binary_label(df)


def prepare_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, list]:
    """Backward-compatible wrapper around shared preprocessing utility."""
    return _prepare_train_test(df)


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.36,
    n_estimators: int = 200,
    random_state: int | None = 42,
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
    """Backward-compatible wrapper around shared evaluation utility."""
    return evaluate_binary(y_true, y_pred)


def run_baseline(
    data_path: Path,
    row_limit: int | None = None,
    contamination: float = 0.36,
    random_state: int | None = 42,
    report_path: Optional[Path] = None,
    results_output_path: Optional[Path] = None,
) -> dict:
    """Execute the full pipeline and print evaluation outputs."""
    print(f"Loading data from {data_path} (row_limit={row_limit})")
    df = load_data(data_path, row_limit=row_limit)
    df = add_binary_label(df)
    label_dist = df["label"].value_counts(normalize=True).mul(100).round(2)
    print("Label distribution (%):")
    print(label_dist.to_string())

    X_train, X_test, y_test, feature_cols = prepare_train_test(df)
    print(f"Using {len(feature_cols)} features -> train: {X_train.shape}, test: {X_test.shape}")

    iso = train_isolation_forest(
        X_train,
        contamination=contamination,
        random_state=random_state,
    )
    raw_pred = iso.predict(X_test.to_numpy())
    y_pred = np.where(raw_pred == 1, 0, 1)
    anomaly_score = -iso.decision_function(X_test.to_numpy())

    cm, report = evaluate(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    if report_path:
        save_baseline_report(report_path, label_dist=label_dist, cm=cm, report=report)
        print(f"\nSaved baseline report to {report_path}")

    if results_output_path:
        results_output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = X_test.copy()
        results_df["true_label"] = y_test.to_numpy()
        results_df["pred_label"] = y_pred
        results_df["anomaly_score"] = anomaly_score
        results_df.to_csv(results_output_path, index_label="index")
        print(f"Baseline results saved to {results_output_path}")

    metrics = compute_metrics_dict(y_true=y_test.to_numpy(), y_pred=y_pred)
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "report_text": report,
        "label_distribution": label_dist,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline Isolation Forest anomaly detector")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/cicids2017_cleaned.csv"),
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument(
        "--results-output-path",
        type=Path,
        default=Path("reports/tables/if_mixed_results.csv"),
        help="Output CSV for per-row baseline predictions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(
        data_path=args.data_path,
        row_limit=args.row_limit,
        contamination=args.contamination,
        random_state=args.seed,
        report_path=args.report_path,
        results_output_path=args.results_output_path,
    )

