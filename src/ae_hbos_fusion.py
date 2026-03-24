"""Score-level fusion between Autoencoder and HBOS on aligned split."""

from __future__ import annotations

import argparse
import json
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


def _select_best(df: pd.DataFrame, min_recall: float, max_fpr: float) -> pd.Series:
    eligible = df[(df["attack_recall"] >= min_recall) & (df["false_positive_rate"] <= max_fpr)]
    if eligible.empty:
        eligible = df[df["attack_recall"] >= min_recall]
    if eligible.empty:
        eligible = df
    return eligible.sort_values(["attack_f1", "false_positive_rate"], ascending=[False, True]).iloc[0]


def _split_protocol(df: pd.DataFrame, seed: int | None):
    feature_cols = [c for c in df.columns if c not in {"Attack Type", "label"}]
    df_normal = df[df["label"] == 0].copy()
    df_attack = df[df["label"] == 1].copy()

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
    )


def run_fusion(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = Path(cfg.get("data_path", "data/cicids2017_cleaned.csv"))
    row_limit = cfg.get("row_limit", 250000)
    seed = cfg.get("seed", 42)
    output_dir = Path(cfg.get("output_dir", "reports/tables"))
    weight_grid = [float(x) for x in cfg.get("weight_grid", [0.2, 0.4, 0.6, 0.8])]
    threshold_percentiles = [float(x) for x in cfg.get("threshold_percentiles", [50, 60, 70, 80, 90, 95, 97, 99])]
    min_recall = float(cfg.get("min_recall", 0.6))
    max_fpr = float(cfg.get("max_fpr", 0.35))
    ae_epochs = int(cfg.get("ae_epochs", 20))
    ae_batch_size = int(cfg.get("ae_batch_size", 512))
    ae_lr = float(cfg.get("ae_lr", 1e-3))
    hbos_bins = int(cfg.get("hbos_bins", 10))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)

    df = add_binary_label(load_data(data_path, row_limit=row_limit))
    X_train, X_val, y_val, X_test, y_test = _split_protocol(df, seed=seed)

    # HBOS scores
    hb = HBOS(n_bins=hbos_bins)
    hb.fit(X_train.to_numpy())
    hb_train_scores = hb.decision_scores_
    hb_val_scores = hb.decision_function(X_val.to_numpy())
    hb_test_scores = hb.decision_function(X_test.to_numpy())

    # AE scores
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    X_val_scaled = scaler.transform(X_val.to_numpy())
    X_test_scaled = scaler.transform(X_test.to_numpy())

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    loader_kwargs = {"batch_size": ae_batch_size, "shuffle": True}
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        loader_kwargs["generator"] = gen
    loader = DataLoader(TensorDataset(train_tensor), **loader_kwargs)

    ae = Autoencoder(input_dim=X_train_scaled.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = ae.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=ae_lr)
    ae.train()
    for _ in range(ae_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = ae(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()

    ae.eval()
    with torch.no_grad():
        recon_train = ae(train_tensor.to(device)).cpu().numpy()
        recon_val = ae(val_tensor.to(device)).cpu().numpy()
        recon_test = ae(test_tensor.to(device)).cpu().numpy()
    ae_train_scores = np.mean((recon_train - X_train_scaled) ** 2, axis=1)
    ae_val_scores = np.mean((recon_val - X_val_scaled) ** 2, axis=1)
    ae_test_scores = np.mean((recon_test - X_test_scaled) ** 2, axis=1)

    hb_train_n, hb_val_n = _normalize(hb_train_scores, hb_val_scores)
    _, hb_test_n = _normalize(hb_train_scores, hb_test_scores)
    ae_train_n, ae_val_n = _normalize(ae_train_scores, ae_val_scores)
    _, ae_test_n = _normalize(ae_train_scores, ae_test_scores)

    candidates = []
    for w_hb in weight_grid:
        val_score = w_hb * hb_val_n + (1.0 - w_hb) * ae_val_n
        train_score = w_hb * hb_train_n + (1.0 - w_hb) * ae_train_n
        for perc in threshold_percentiles:
            thr = float(np.percentile(train_score, perc))
            pred_val = (val_score > thr).astype(int)
            metrics = compute_metrics_dict(y_val.to_numpy(), pred_val)
            candidates.append(
                {
                    "weight_hbos": w_hb,
                    "weight_ae": 1.0 - w_hb,
                    "threshold_percentile": perc,
                    "threshold_value": thr,
                    **metrics,
                }
            )

    cand_df = pd.DataFrame(candidates).sort_values(["attack_f1", "false_positive_rate"], ascending=[False, True]).reset_index(drop=True)
    best = _select_best(cand_df, min_recall=min_recall, max_fpr=max_fpr)

    test_score = best["weight_hbos"] * hb_test_n + best["weight_ae"] * ae_test_n
    test_pred = (test_score > float(best["threshold_value"])).astype(int)
    test_metrics = compute_metrics_dict(y_test.to_numpy(), test_pred)

    out_df = X_test.copy()
    out_df["true_label"] = y_test.to_numpy()
    out_df["pred_label"] = test_pred
    out_df["ae_score"] = ae_test_scores
    out_df["hbos_score"] = hb_test_scores
    out_df["fusion_score"] = test_score
    out_df["anomaly_score"] = test_score
    out_df.to_csv(output_dir / "ae_hbos_fusion_results.csv", index_label="index")
    cand_df.to_csv(output_dir / "ae_hbos_fusion_candidates.csv", index=False)

    fusion_test_row = {
        "model": "AE+HBOS Fusion",
        "selected_threshold_percentile": float(best["threshold_percentile"]),
        "selected_threshold_value": float(best["threshold_value"]),
        **test_metrics,
    }
    pd.DataFrame([fusion_test_row]).to_csv(output_dir / "ae_hbos_fusion_test_metrics.csv", index=False)

    summary = {
        "best_config": best.to_dict(),
        "test_metrics": test_metrics,
        "settings": {
            "data_path": str(data_path),
            "row_limit": row_limit,
            "seed": seed,
            "hbos_bins": hbos_bins,
            "ae_epochs": ae_epochs,
            "ae_batch_size": ae_batch_size,
            "ae_lr": ae_lr,
            "min_recall": min_recall,
            "max_fpr": max_fpr,
            "weight_grid": weight_grid,
            "threshold_percentiles": threshold_percentiles,
        },
        "artifacts": {
            "candidates": str(output_dir / "ae_hbos_fusion_candidates.csv"),
            "results": str(output_dir / "ae_hbos_fusion_results.csv"),
            "test_metrics": str(output_dir / "ae_hbos_fusion_test_metrics.csv"),
        },
    }
    (output_dir / "ae_hbos_fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("AE+HBOS fusion completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AE + HBOS score fusion")
    parser.add_argument("--config-path", type=Path, default=Path("config/ae_hbos_fusion.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fusion(config_path=args.config_path)
