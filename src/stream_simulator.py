"""
Replay-based streaming utilities for dashboard demos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

MODEL_RESULT_FILES = {
    "Isolation Forest": "if_mixed_results.csv",
    "Autoencoder": "ae_mixed_results.csv",
    "Hybrid": "hybrid_mixed_results.csv",
    "PyOD-HBOS": "pyod_hbos_mixed_results.csv",
    "AE+HBOS Fusion": "ae_hbos_fusion_results.csv",
}


def resolve_results_path(model_name: str, tables_dir: Path = Path("reports/tables")) -> Path:
    if model_name not in MODEL_RESULT_FILES:
        raise ValueError(f"Unsupported model name: {model_name}")
    return tables_dir / MODEL_RESULT_FILES[model_name]


def load_results_dataframe(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    required = {"true_label", "pred_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results file missing required columns: {sorted(missing)}")

    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = df["pred_label"].astype(float)
    return df


def replay_slice(df: pd.DataFrame, cursor: int, batch_size: int) -> tuple[pd.DataFrame, int]:
    new_cursor = min(cursor + batch_size, len(df))
    return df.iloc[:new_cursor].copy(), new_cursor


def compute_live_metrics(df_live: pd.DataFrame, window_size: int = 200) -> dict:
    if df_live.empty:
        return {
            "total_rows": 0,
            "anomaly_count": 0,
            "anomaly_ratio": 0.0,
            "confusion_matrix": np.array([[0, 0], [0, 0]]),
            "recent_anomalies": df_live,
            "trend_df": pd.DataFrame(columns=["window", "anomaly_ratio"]),
        }

    total_rows = len(df_live)
    anomaly_count = int((df_live["pred_label"] == 1).sum())
    anomaly_ratio = float(anomaly_count / total_rows)
    cm = confusion_matrix(df_live["true_label"], df_live["pred_label"], labels=[0, 1])

    recent_anomalies = df_live[df_live["pred_label"] == 1].tail(20)

    idx = np.arange(total_rows)
    trend = (
        pd.DataFrame({"window": idx // window_size, "pred_label": df_live["pred_label"].to_numpy()})
        .groupby("window")
        .agg(anomaly_ratio=("pred_label", "mean"))
        .reset_index()
    )

    return {
        "total_rows": total_rows,
        "anomaly_count": anomaly_count,
        "anomaly_ratio": anomaly_ratio,
        "confusion_matrix": cm,
        "recent_anomalies": recent_anomalies,
        "trend_df": trend,
    }
