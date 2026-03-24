"""
PyOD benchmark runner on fair train_normal / val_mixed / test_mixed protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from sklearn.model_selection import train_test_split

from data_preprocessing import add_binary_label, load_data
from evaluate import compute_metrics_dict


def _split_protocol(df: pd.DataFrame, seed: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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


def _select_best(candidates: pd.DataFrame, min_recall: float, max_fpr: float) -> pd.Series:
    eligible = candidates[(candidates["attack_recall"] >= min_recall) & (candidates["false_positive_rate"] <= max_fpr)]
    if eligible.empty:
        eligible = candidates[candidates["attack_recall"] >= min_recall]
    if eligible.empty:
        eligible = candidates
    return eligible.sort_values(["attack_f1", "false_positive_rate"], ascending=[False, True]).iloc[0]


def _build_model(name: str):
    name = name.upper()
    if name == "ECOD":
        return ECOD()
    if name == "COPOD":
        return COPOD()
    if name == "HBOS":
        return HBOS()
    if name == "LOF":
        return LOF()
    raise ValueError(f"Unsupported PyOD model: {name}")


def _apply_feature_transform(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    transform: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if transform == "none":
        return X_train, X_val, X_test
    if transform == "log1p":
        def tx(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            return np.log1p(np.clip(out, a_min=0.0, a_max=None))

        return tx(X_train), tx(X_val), tx(X_test)
    raise ValueError(f"Unsupported feature transform: {transform}")


def _evaluate_scores(
    model_label: str,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_val: pd.Series,
    y_test: pd.Series,
    threshold_percentiles: list[float],
    min_recall: float,
    max_fpr: float,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    local_rows = []
    for perc in threshold_percentiles:
        thr = float(pd.Series(train_scores).quantile(perc / 100.0))
        y_pred_val = (val_scores > thr).astype(int)
        metrics = compute_metrics_dict(y_val.to_numpy(), y_pred_val)
        local_rows.append({"model": model_label, "threshold_percentile": perc, "threshold_value": thr, **metrics})

    local_df = pd.DataFrame(local_rows)
    best = _select_best(local_df, min_recall=min_recall, max_fpr=max_fpr)
    y_pred_test = (test_scores > float(best["threshold_value"])).astype(int)
    test_metrics = compute_metrics_dict(y_test.to_numpy(), y_pred_test)
    results_df = pd.DataFrame({"true_label": y_test.to_numpy(), "pred_label": y_pred_test, "anomaly_score": test_scores})
    test_row = {
        "model": f"PyOD-{model_label}",
        "selected_threshold_percentile": float(best["threshold_percentile"]),
        "selected_threshold_value": float(best["threshold_value"]),
        **test_metrics,
    }
    return local_df, test_row, results_df


def run_pyod_benchmark(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = Path(cfg.get("data_path", "data/cicids2017_cleaned.csv"))
    row_limit = cfg.get("row_limit")
    seed = cfg.get("seed")
    min_recall = float(cfg.get("min_recall", 0.6))
    max_fpr = float(cfg.get("max_fpr", 0.35))
    models = [str(m).upper() for m in cfg.get("models", ["ECOD", "COPOD", "HBOS"])]
    threshold_percentiles = [float(x) for x in cfg.get("threshold_percentiles", [50, 60, 70, 80, 90, 95, 97, 99])]
    feature_transform = str(cfg.get("feature_transform", "none"))
    hbos_bins_grid = [int(x) for x in cfg.get("hbos_bins_grid", [10])]
    hbos_ensemble_bins = [int(x) for x in cfg.get("hbos_ensemble_bins", [])]
    output_dir = Path(cfg.get("output_dir", "reports/tables"))
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_binary_label(load_data(data_path, row_limit=row_limit))
    X_train, X_val, y_val, X_test, y_test = _split_protocol(df, seed=seed)
    X_train, X_val, X_test = _apply_feature_transform(X_train, X_val, X_test, transform=feature_transform)

    val_rows: list[dict] = []
    test_rows: list[dict] = []
    hbos_result_candidates: list[tuple[float, pd.DataFrame]] = []

    for model_name in models:
        if model_name != "HBOS":
            model = _build_model(model_name)
            model.fit(X_train.to_numpy())
            local_df, test_row, results_df = _evaluate_scores(
                model_label=model_name,
                train_scores=model.decision_scores_,
                val_scores=model.decision_function(X_val.to_numpy()),
                test_scores=model.decision_function(X_test.to_numpy()),
                y_val=y_val,
                y_test=y_test,
                threshold_percentiles=threshold_percentiles,
                min_recall=min_recall,
                max_fpr=max_fpr,
            )
            val_rows.extend(local_df.to_dict(orient="records"))
            test_rows.append(test_row)
            results_df.to_csv(output_dir / f"pyod_{model_name.lower()}_mixed_results.csv", index_label="index")
            continue

        # HBOS with bin tuning
        for n_bins in hbos_bins_grid:
            hbos = HBOS(n_bins=n_bins)
            hbos.fit(X_train.to_numpy())
            label = f"HBOS-bins{n_bins}"
            local_df, test_row, results_df = _evaluate_scores(
                model_label=label,
                train_scores=hbos.decision_scores_,
                val_scores=hbos.decision_function(X_val.to_numpy()),
                test_scores=hbos.decision_function(X_test.to_numpy()),
                y_val=y_val,
                y_test=y_test,
                threshold_percentiles=threshold_percentiles,
                min_recall=min_recall,
                max_fpr=max_fpr,
            )
            val_rows.extend(local_df.to_dict(orient="records"))
            test_rows.append(test_row)
            results_df.to_csv(output_dir / f"pyod_hbos_bins{n_bins}_mixed_results.csv", index_label="index")
            hbos_result_candidates.append((float(test_row["attack_f1"]), results_df))

        # Optional HBOS ensemble across bin settings
        if hbos_ensemble_bins:
            train_stack = []
            val_stack = []
            test_stack = []
            for n_bins in hbos_ensemble_bins:
                h = HBOS(n_bins=n_bins)
                h.fit(X_train.to_numpy())
                train_stack.append(h.decision_scores_)
                val_stack.append(h.decision_function(X_val.to_numpy()))
                test_stack.append(h.decision_function(X_test.to_numpy()))
            train_scores = np.median(np.vstack(train_stack), axis=0)
            val_scores = np.median(np.vstack(val_stack), axis=0)
            test_scores = np.median(np.vstack(test_stack), axis=0)
            local_df, test_row, results_df = _evaluate_scores(
                model_label="HBOS-ensemble",
                train_scores=train_scores,
                val_scores=val_scores,
                test_scores=test_scores,
                y_val=y_val,
                y_test=y_test,
                threshold_percentiles=threshold_percentiles,
                min_recall=min_recall,
                max_fpr=max_fpr,
            )
            val_rows.extend(local_df.to_dict(orient="records"))
            test_rows.append(test_row)
            results_df.to_csv(output_dir / "pyod_hbos_ensemble_mixed_results.csv", index_label="index")
            hbos_result_candidates.append((float(test_row["attack_f1"]), results_df))

    val_df = pd.DataFrame(val_rows).sort_values(["attack_f1", "false_positive_rate"], ascending=[False, True]).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sort_values(["attack_f1", "false_positive_rate"], ascending=[False, True]).reset_index(drop=True)
    val_df.to_csv(output_dir / "pyod_validation_sweep.csv", index=False)
    test_df.to_csv(output_dir / "pyod_model_comparison_test.csv", index=False)

    # Expose strongest HBOS variant for dashboard replay.
    if hbos_result_candidates:
        best_hbos_df = sorted(hbos_result_candidates, key=lambda x: x[0], reverse=True)[0][1]
        best_hbos_df.to_csv(output_dir / "pyod_hbos_mixed_results.csv", index_label="index")

    # If core comparison exists, merge for one-table view.
    core_path = output_dir / "experiment_model_comparison_test.csv"
    fusion_path = output_dir / "ae_hbos_fusion_test_metrics.csv"
    if core_path.exists():
        core_df = pd.read_csv(core_path)
        frames = [core_df, test_df]
        if fusion_path.exists():
            frames.append(pd.read_csv(fusion_path))
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_dir / "model_comparison_with_pyod.csv", index=False)

    report_lines = [
        "# PyOD Benchmark Report",
        "",
        "## Protocol",
        "- train_normal / val_mixed / test_mixed",
        "- label mapping: normal=0, attack=1",
        f"- models: {models}",
        f"- feature_transform: {feature_transform}",
        f"- hbos_bins_grid: {hbos_bins_grid}",
        f"- hbos_ensemble_bins: {hbos_ensemble_bins}",
        f"- min_recall: {min_recall}",
        f"- max_fpr: {max_fpr}",
        "",
        "## Outputs",
        f"- `{output_dir / 'pyod_validation_sweep.csv'}`",
        f"- `{output_dir / 'pyod_model_comparison_test.csv'}`",
    ]
    if (output_dir / "model_comparison_with_pyod.csv").exists():
        report_lines.append(f"- `{output_dir / 'model_comparison_with_pyod.csv'}`")
    (output_dir / "pyod_benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("PyOD benchmark completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyOD benchmark")
    parser.add_argument("--config-path", type=Path, default=Path("config/pyod_benchmark.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pyod_benchmark(config_path=args.config_path)
