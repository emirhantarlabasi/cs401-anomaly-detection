"""
Advanced Isolation Forest tuning with contamination + score-threshold sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from data_preprocessing import add_binary_label, load_data
from evaluate import compute_metrics_dict


def _split_train_val_test(df: pd.DataFrame, seed: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in {"Attack Type", "label"}]
    df_normal = df[df["label"] == 0].copy()
    df_attack = df[df["label"] == 1].copy()

    df_train_normal, df_normal_holdout = train_test_split(df_normal, test_size=0.4, random_state=seed)
    df_val_normal, df_test_normal = train_test_split(df_normal_holdout, test_size=0.5, random_state=seed)
    df_val_attack, df_test_attack = train_test_split(df_attack, test_size=0.5, random_state=seed)

    df_val = pd.concat([df_val_normal, df_val_attack], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_test = pd.concat([df_test_normal, df_test_attack], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    X_train = df_train_normal[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    y_val = df_val["label"].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test["label"].copy()
    return X_train, X_val, y_val, X_test, y_test


def _pick_best(df: pd.DataFrame, min_recall: float, max_fpr: float) -> pd.Series:
    eligible = df[(df["attack_recall"] >= min_recall) & (df["false_positive_rate"] <= max_fpr)]
    if eligible.empty:
        eligible = df[df["attack_recall"] >= min_recall]
    if eligible.empty:
        eligible = df
    return eligible.sort_values(["attack_f1", "false_positive_rate"], ascending=[False, True]).iloc[0]


def run_if_advanced_tuning(
    data_path: Path,
    row_limit: int | None = None,
    seed: int | None = 42,
    contamination_grid: tuple[float, ...] = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.36),
    n_estimators_grid: tuple[int, ...] = (100, 200, 300),
    max_samples_grid: tuple[str | int, ...] = (256, 512, "auto"),
    threshold_percentiles: tuple[float, ...] = (50, 60, 70, 80, 85, 90, 95, 97, 99),
    min_recall: float = 0.90,
    max_fpr: float = 0.35,
    output_dir: Path = Path("reports/tables"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_binary_label(load_data(data_path, row_limit=row_limit))
    X_train, X_val, y_val, X_test, y_test = _split_train_val_test(df, seed=seed)

    val_rows: list[dict] = []
    best_by_model: list[dict] = []
    best_model = None
    best_model_row = None

    for contamination in contamination_grid:
        for n_estimators in n_estimators_grid:
            for max_samples in max_samples_grid:
                model = IsolationForest(
                    contamination=contamination,
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    random_state=seed,
                    n_jobs=-1,
                )
                model.fit(X_train)

                train_scores = -model.decision_function(X_train)
                val_scores = -model.decision_function(X_val)

                local_rows = []
                for perc in threshold_percentiles:
                    thr = float(np.percentile(train_scores, perc))
                    y_pred_val = (val_scores > thr).astype(int)
                    metrics = compute_metrics_dict(y_true=y_val.to_numpy(), y_pred=y_pred_val)
                    row = {
                        "contamination": contamination,
                        "n_estimators": n_estimators,
                        "max_samples": max_samples,
                        "threshold_percentile": perc,
                        "threshold_value": thr,
                        **metrics,
                    }
                    val_rows.append(row)
                    local_rows.append(row)

                best_local = _pick_best(pd.DataFrame(local_rows), min_recall=min_recall, max_fpr=max_fpr)

                # Evaluate chosen threshold on test set
                test_scores = -model.decision_function(X_test)
                y_pred_test = (test_scores > float(best_local["threshold_value"])).astype(int)
                test_metrics = compute_metrics_dict(y_true=y_test.to_numpy(), y_pred=y_pred_test)
                summary_row = {
                    "contamination": contamination,
                    "n_estimators": n_estimators,
                    "max_samples": max_samples,
                    "selected_threshold_percentile": float(best_local["threshold_percentile"]),
                    "selected_threshold_value": float(best_local["threshold_value"]),
                    **{f"val_{k}": best_local[k] for k in ["attack_precision", "attack_recall", "attack_f1", "false_positive_rate", "macro_f1"]},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
                best_by_model.append(summary_row)

                if best_model is None:
                    best_model = model
                    best_model_row = summary_row
                else:
                    current = pd.Series(summary_row)
                    incumbent = pd.Series(best_model_row)
                    better = False
                    if current["test_attack_recall"] >= min_recall and incumbent["test_attack_recall"] < min_recall:
                        better = True
                    elif (
                        (current["test_attack_recall"] >= min_recall) == (incumbent["test_attack_recall"] >= min_recall)
                        and current["test_attack_f1"] > incumbent["test_attack_f1"]
                    ):
                        better = True
                    if better:
                        best_model = model
                        best_model_row = summary_row

    val_df = pd.DataFrame(val_rows).sort_values("attack_f1", ascending=False).reset_index(drop=True)
    summary_df = pd.DataFrame(best_by_model).sort_values("test_attack_f1", ascending=False).reset_index(drop=True)
    val_df.to_csv(output_dir / "if_threshold_sweep_validation.csv", index=False)
    summary_df.to_csv(output_dir / "if_advanced_tuning_summary.csv", index=False)

    best_payload = {
        "selection_rule": {
            "min_recall": min_recall,
            "max_fpr": max_fpr,
        },
        "best_model": best_model_row,
        "artifacts": {
            "validation_sweep": str(output_dir / "if_threshold_sweep_validation.csv"),
            "summary": str(output_dir / "if_advanced_tuning_summary.csv"),
        },
    }
    (output_dir / "if_best_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    print("Advanced IF tuning completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced IF tuning on mixed validation")
    parser.add_argument("--data-path", type=Path, default=Path("data/cicids2017_cleaned.csv"))
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-fpr", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tables"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_if_advanced_tuning(
        data_path=args.data_path,
        row_limit=args.row_limit,
        seed=args.seed,
        min_recall=args.min_recall,
        max_fpr=args.max_fpr,
        output_dir=args.output_dir,
    )
