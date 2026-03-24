"""
Evaluation helpers shared by model scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_binary(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[np.ndarray, str]:
    """Compute confusion matrix and text report for binary classification."""
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3)
    return cm, report


def save_baseline_report(report_path: Path, label_dist: pd.Series, cm: np.ndarray, report: str) -> None:
    """Persist baseline evaluation in a simple human-readable txt format."""
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


def compute_metrics_dict(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return normalized metrics used by benchmark tables."""
    report_dict = classification_report(y_true, y_pred, digits=3, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "accuracy": report_dict["accuracy"],
        "attack_precision": report_dict["1"]["precision"],
        "attack_recall": report_dict["1"]["recall"],
        "attack_f1": report_dict["1"]["f1-score"],
        "normal_recall": report_dict["0"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "false_positive_rate": fpr,
    }


def save_benchmark_table(rows: dict[str, dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Save benchmark metrics to CSV and return dataframe."""
    df = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="model")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
