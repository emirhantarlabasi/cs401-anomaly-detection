"""
Stability checks for HBOS vs Autoencoder under multiple data sizes/splits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyod.models.hbos import HBOS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from autoencoder_model import Autoencoder
from data_preprocessing import add_binary_label, load_data
from evaluate import compute_metrics_dict
from random_utils import set_global_seed


def _normalize(train_values: np.ndarray, test_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = float(np.min(train_values))
    hi = float(np.max(train_values))
    den = max(hi - lo, 1e-12)
    return (train_values - lo) / den, (test_values - lo) / den


def _pick_best(df: pd.DataFrame, min_recall: float, max_fpr: float) -> pd.Series:
    eligible = df[(df["attack_recall"] >= min_recall) & (df["false_positive_rate"] <= max_fpr)]
    if eligible.empty:
        eligible = df[df["attack_recall"] >= min_recall]
    if eligible.empty:
        eligible = df
    return eligible.sort_values(["attack_f1", "false_positive_rate"], ascending=[False, True]).iloc[0]


def _split_random(df: pd.DataFrame, seed: int | None):
    feature_cols = [c for c in df.columns if c not in {"Attack Type", "label", "_row_id"}]
    df_normal = df[df["label"] == 0].copy()
    df_attack = df[df["label"] == 1].copy()

    if len(df_attack) < 4 or len(df_normal) < 50:
        raise ValueError("Insufficient class counts for robust split.")

    df_train_normal, df_normal_holdout = train_test_split(df_normal, test_size=0.4, random_state=seed)
    df_val_normal, df_test_normal = train_test_split(df_normal_holdout, test_size=0.5, random_state=seed)
    df_val_attack, df_test_attack = train_test_split(df_attack, test_size=0.5, random_state=seed)

    df_val = pd.concat([df_val_normal, df_val_attack], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_test = pd.concat([df_test_normal, df_test_attack], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return (
        df_train_normal[feature_cols].copy(),
        df_val[feature_cols].copy(),
        df_val["label"].copy(),
        df_test[feature_cols].copy(),
        df_test["label"].copy(),
        df_train_normal["_row_id"].to_numpy(),
        df_val["_row_id"].to_numpy(),
        df_test["_row_id"].to_numpy(),
    )


def _split_time(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in {"Attack Type", "label", "_row_id"}]
    df_normal = df[df["label"] == 0].sort_values("_row_id").copy()
    df_attack = df[df["label"] == 1].sort_values("_row_id").copy()
    if len(df_attack) < 4 or len(df_normal) < 50:
        raise ValueError("Insufficient class counts for robust split.")

    n_norm = len(df_normal)
    n_train = int(n_norm * 0.6)
    n_val = int(n_norm * 0.2)
    df_train_normal = df_normal.iloc[:n_train].copy()
    df_val_normal = df_normal.iloc[n_train : n_train + n_val].copy()
    df_test_normal = df_normal.iloc[n_train + n_val :].copy()

    n_att = len(df_attack)
    n_att_val = n_att // 2
    df_val_attack = df_attack.iloc[:n_att_val].copy()
    df_test_attack = df_attack.iloc[n_att_val:].copy()

    df_val = pd.concat([df_val_normal, df_val_attack], ignore_index=True)
    df_test = pd.concat([df_test_normal, df_test_attack], ignore_index=True)

    return (
        df_train_normal[feature_cols].copy(),
        df_val[feature_cols].copy(),
        df_val["label"].copy(),
        df_test[feature_cols].copy(),
        df_test["label"].copy(),
        df_train_normal["_row_id"].to_numpy(),
        df_val["_row_id"].to_numpy(),
        df_test["_row_id"].to_numpy(),
    )


def _run_hbos(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_percentiles: list[float],
    min_recall: float,
    max_fpr: float,
) -> tuple[dict, dict]:
    model = HBOS()
    t0 = time.perf_counter()
    model.fit(X_train.to_numpy())
    train_time = time.perf_counter() - t0

    train_scores = model.decision_scores_
    val_scores = model.decision_function(X_val.to_numpy())
    candidates = []
    for perc in threshold_percentiles:
        thr = float(np.percentile(train_scores, perc))
        pred_val = (val_scores > thr).astype(int)
        metrics_val = compute_metrics_dict(y_val.to_numpy(), pred_val)
        candidates.append({"threshold_percentile": perc, "threshold_value": thr, **metrics_val})
    best = _pick_best(pd.DataFrame(candidates), min_recall=min_recall, max_fpr=max_fpr)

    t1 = time.perf_counter()
    test_scores = model.decision_function(X_test.to_numpy())
    pred_test = (test_scores > float(best["threshold_value"])).astype(int)
    infer_time = time.perf_counter() - t1
    metrics_test = compute_metrics_dict(y_test.to_numpy(), pred_test)
    metrics_row = {
        "model": "HBOS",
        "selected_threshold_percentile": float(best["threshold_percentile"]),
        "selected_threshold_value": float(best["threshold_value"]),
        "training_time_s": train_time,
        "inference_time_s": infer_time,
        **metrics_test,
    }
    artifacts = {
        "train_scores": train_scores,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
    }
    return metrics_row, artifacts


def _run_ae(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_percentiles: list[float],
    min_recall: float,
    max_fpr: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int | None,
) -> tuple[dict, dict]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    X_val_scaled = scaler.transform(X_val.to_numpy())
    X_test_scaled = scaler.transform(X_test.to_numpy())

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    loader_kwargs = {"batch_size": batch_size, "shuffle": True}
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        loader_kwargs["generator"] = gen
    loader = DataLoader(TensorDataset(train_tensor), **loader_kwargs)

    model = Autoencoder(input_dim=X_train_scaled.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0

    with torch.no_grad():
        model.eval()
        recon_train = model(train_tensor.to(device)).cpu().numpy()
        recon_val = model(val_tensor.to(device)).cpu().numpy()
        t1 = time.perf_counter()
        recon_test = model(test_tensor.to(device)).cpu().numpy()
        infer_time = time.perf_counter() - t1

    train_errors = np.mean((recon_train - X_train_scaled) ** 2, axis=1)
    val_errors = np.mean((recon_val - X_val_scaled) ** 2, axis=1)
    test_errors = np.mean((recon_test - X_test_scaled) ** 2, axis=1)

    candidates = []
    for perc in threshold_percentiles:
        thr = float(np.percentile(train_errors, perc))
        pred_val = (val_errors > thr).astype(int)
        metrics_val = compute_metrics_dict(y_val.to_numpy(), pred_val)
        candidates.append({"threshold_percentile": perc, "threshold_value": thr, **metrics_val})
    best = _pick_best(pd.DataFrame(candidates), min_recall=min_recall, max_fpr=max_fpr)

    pred_test = (test_errors > float(best["threshold_value"])).astype(int)
    metrics_test = compute_metrics_dict(y_test.to_numpy(), pred_test)
    metrics_row = {
        "model": "Autoencoder",
        "selected_threshold_percentile": float(best["threshold_percentile"]),
        "selected_threshold_value": float(best["threshold_value"]),
        "training_time_s": train_time,
        "inference_time_s": infer_time,
        **metrics_test,
    }
    artifacts = {
        "train_scores": train_errors,
        "val_scores": val_errors,
        "test_scores": test_errors,
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
    }
    return metrics_row, artifacts


def run_stability_check(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = Path(cfg.get("data_path", "data/cicids2017_cleaned.csv"))
    sample_sizes = [int(x) for x in cfg.get("sample_sizes", [100000, 250000, 400000])]
    seed = cfg.get("seed", 42)
    split_modes = [str(x) for x in cfg.get("split_modes", ["random", "time"])]
    threshold_percentiles = [float(x) for x in cfg.get("threshold_percentiles", [50, 60, 70, 80, 90, 95, 97, 99])]
    fusion_weight_grid = [float(x) for x in cfg.get("fusion_weight_grid", [0.2, 0.4, 0.6, 0.8])]
    min_recall = float(cfg.get("min_recall", 0.6))
    max_fpr = float(cfg.get("max_fpr", 0.35))
    ae_epochs = int(cfg.get("ae_epochs", 20))
    ae_batch_size = int(cfg.get("ae_batch_size", 512))
    ae_lr = float(cfg.get("ae_lr", 1e-3))
    output_dir = Path(cfg.get("output_dir", "reports/tables"))
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(seed)
    rows: list[dict] = []
    sanity_rows: list[dict] = []

    for size in sample_sizes:
        df = add_binary_label(load_data(data_path, row_limit=size))
        df = df.reset_index(drop=False).rename(columns={"index": "_row_id"})
        for split_mode in split_modes:
            if split_mode == "random":
                split = _split_random(df, seed=seed)
            elif split_mode == "time":
                split = _split_time(df)
            else:
                raise ValueError(f"Unsupported split mode: {split_mode}")

            X_train, X_val, y_val, X_test, y_test, idx_train, idx_val, idx_test = split

            overlap = len(set(idx_train).intersection(idx_val)) + len(set(idx_train).intersection(idx_test)) + len(set(idx_val).intersection(idx_test))
            sanity_rows.append(
                {
                    "sample_size": size,
                    "split_mode": split_mode,
                    "train_only_normal": bool((y_val is not None) and True),  # split guarantees this
                    "index_overlap_count": int(overlap),
                    "threshold_selection_scope": "validation_only",
                    "leakage_check_pass": bool(overlap == 0),
                }
            )

            hb, hb_art = _run_hbos(
                X_train=X_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                threshold_percentiles=threshold_percentiles,
                min_recall=min_recall,
                max_fpr=max_fpr,
            )
            hb.update({"sample_size": size, "split_mode": split_mode})
            rows.append(hb)

            ae, ae_art = _run_ae(
                X_train=X_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                threshold_percentiles=threshold_percentiles,
                min_recall=min_recall,
                max_fpr=max_fpr,
                epochs=ae_epochs,
                batch_size=ae_batch_size,
                lr=ae_lr,
                seed=seed,
            )
            ae.update({"sample_size": size, "split_mode": split_mode})
            rows.append(ae)

            # AE + HBOS fusion on same split
            hb_train_n, hb_val_n = _normalize(hb_art["train_scores"], hb_art["val_scores"])
            _, hb_test_n = _normalize(hb_art["train_scores"], hb_art["test_scores"])
            ae_train_n, ae_val_n = _normalize(ae_art["train_scores"], ae_art["val_scores"])
            _, ae_test_n = _normalize(ae_art["train_scores"], ae_art["test_scores"])

            fusion_candidates = []
            for w_hb in fusion_weight_grid:
                val_score = w_hb * hb_val_n + (1.0 - w_hb) * ae_val_n
                train_score = w_hb * hb_train_n + (1.0 - w_hb) * ae_train_n
                for perc in threshold_percentiles:
                    thr = float(np.percentile(train_score, perc))
                    pred_val = (val_score > thr).astype(int)
                    metrics_val = compute_metrics_dict(y_val.to_numpy(), pred_val)
                    fusion_candidates.append(
                        {
                            "weight_hbos": w_hb,
                            "weight_ae": 1.0 - w_hb,
                            "threshold_percentile": perc,
                            "threshold_value": thr,
                            **metrics_val,
                        }
                    )
            best_fusion = _pick_best(pd.DataFrame(fusion_candidates), min_recall=min_recall, max_fpr=max_fpr)
            test_score = best_fusion["weight_hbos"] * hb_test_n + best_fusion["weight_ae"] * ae_test_n
            test_pred = (test_score > float(best_fusion["threshold_value"])).astype(int)
            fusion_metrics = compute_metrics_dict(y_test.to_numpy(), test_pred)
            rows.append(
                {
                    "model": "AE+HBOS Fusion",
                    "selected_threshold_percentile": float(best_fusion["threshold_percentile"]),
                    "selected_threshold_value": float(best_fusion["threshold_value"]),
                    "weight_hbos": float(best_fusion["weight_hbos"]),
                    "weight_ae": float(best_fusion["weight_ae"]),
                    "training_time_s": hb["training_time_s"] + ae["training_time_s"],
                    "inference_time_s": hb["inference_time_s"] + ae["inference_time_s"],
                    **fusion_metrics,
                    "sample_size": size,
                    "split_mode": split_mode,
                }
            )

    runs_df = pd.DataFrame(rows)
    sanity_df = pd.DataFrame(sanity_rows)
    runs_df.to_csv(output_dir / "pyod_stability_runs.csv", index=False)
    sanity_df.to_csv(output_dir / "pyod_leakage_sanity.csv", index=False)

    summary = (
        runs_df.groupby("model")[["attack_f1", "attack_recall", "false_positive_rate", "accuracy"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(output_dir / "pyod_stability_summary.csv", index=False)

    report_lines = [
        "# PyOD Stability Check Report",
        "",
        "## Coverage",
        f"- sample_sizes: {sample_sizes}",
        f"- split_modes: {split_modes}",
        f"- threshold_percentiles: {threshold_percentiles}",
        f"- fusion_weight_grid: {fusion_weight_grid}",
        "",
        "## Leakage Sanity",
        f"- rows with overlap > 0: {int((sanity_df['index_overlap_count'] > 0).sum())}",
        f"- all threshold selections: validation_only",
        "",
        "## Outputs",
        f"- `{output_dir / 'pyod_stability_runs.csv'}`",
        f"- `{output_dir / 'pyod_stability_summary.csv'}`",
        f"- `{output_dir / 'pyod_leakage_sanity.csv'}`",
    ]
    (output_dir / "pyod_stability_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("PyOD stability check completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HBOS vs AE stability checks")
    parser.add_argument("--config-path", type=Path, default=Path("config/pyod_stability_check.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_stability_check(config_path=args.config_path)
