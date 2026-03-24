"""
Fair model selection runner with fixed train/val/test protocol.
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
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import add_binary_label, load_data
from evaluate import compute_metrics_dict
from random_utils import set_global_seed


class TunableAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def split_protocol(df: pd.DataFrame, seed: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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


def select_best_candidate(candidates: pd.DataFrame, min_recall: float) -> pd.Series:
    eligible = candidates[candidates["attack_recall"] >= min_recall]
    pool = eligible if not eligible.empty else candidates
    pool = pool.sort_values(["attack_f1", "false_positive_rate"], ascending=[False, True])
    return pool.iloc[0]


def run_if_search(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float,
    seed: int | None,
    param_grid: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "contamination": [0.01, 0.03, 0.05, 0.10],
            "max_samples": [256, 512, "auto"],
        }
    rows: list[dict] = []
    for params in ParameterGrid(param_grid):
        t0 = time.perf_counter()
        model = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            max_samples=params["max_samples"],
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train)
        train_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = np.where(model.predict(X_val) == 1, 0, 1)
        infer_s = time.perf_counter() - t1
        metrics = compute_metrics_dict(y_val.to_numpy(), y_pred)
        rows.append({**params, **metrics, "training_time_s": train_s, "inference_time_s": infer_s})

    df = pd.DataFrame(rows).sort_values("attack_f1", ascending=False).reset_index(drop=True)
    best = select_best_candidate(df, min_recall=min_recall)
    return df, best


def train_ae_model(
    X_train: pd.DataFrame,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int | None,
) -> tuple[TunableAutoencoder, StandardScaler, np.ndarray, float]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

    loader_kwargs = {"batch_size": batch_size, "shuffle": True}
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        loader_kwargs["generator"] = gen
    loader = DataLoader(TensorDataset(tensor), **loader_kwargs)

    model = TunableAutoencoder(
        input_dim=X_train_scaled.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
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
    train_s = time.perf_counter() - t0

    with torch.no_grad():
        model.eval()
        recon_train = model(tensor.to(device)).cpu().numpy()
    train_errors = np.mean((recon_train - X_train_scaled) ** 2, axis=1)
    return model, scaler, train_errors, train_s


def predict_ae_scores(model: TunableAutoencoder, scaler: StandardScaler, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    X_scaled = scaler.transform(X.to_numpy())
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    device = next(model.parameters()).device
    t0 = time.perf_counter()
    with torch.no_grad():
        recon = model(tensor.to(device)).cpu().numpy()
    infer_s = time.perf_counter() - t0
    errors = np.mean((recon - X_scaled) ** 2, axis=1)
    return errors, infer_s


def run_ae_search(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float,
    seed: int | None,
    ae_grid: dict | None = None,
    percentile_grid: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    if ae_grid is None:
        ae_grid = {
            "hidden_dim": [64, 128],
            "latent_dim": [16, 32],
            "epochs": [10, 20],
            "batch_size": [512],
            "lr": [1e-3],
        }
    if percentile_grid is None:
        percentile_grid = [95.0, 97.0, 99.0, 99.5]
    rows: list[dict] = []
    model_cache: dict[str, dict] = {}

    for params in ParameterGrid(ae_grid):
        model, scaler, train_errors, train_s = train_ae_model(
            X_train=X_train,
            hidden_dim=params["hidden_dim"],
            latent_dim=params["latent_dim"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            seed=seed,
        )
        val_scores, infer_s = predict_ae_scores(model, scaler, X_val)
        for perc in percentile_grid:
            thr = float(np.percentile(train_errors, perc))
            preds = (val_scores > thr).astype(int)
            metrics = compute_metrics_dict(y_val.to_numpy(), preds)
            row = {
                **params,
                "threshold_percentile": perc,
                "threshold_value": thr,
                **metrics,
                "training_time_s": train_s,
                "inference_time_s": infer_s,
            }
            rows.append(row)

        key = json.dumps(params, sort_keys=True)
        model_cache[key] = {"model": model, "scaler": scaler, "train_errors": train_errors}

    df = pd.DataFrame(rows).sort_values("attack_f1", ascending=False).reset_index(drop=True)
    best = select_best_candidate(df, min_recall=min_recall)
    best_key = json.dumps(
        {
            "hidden_dim": int(best["hidden_dim"]),
            "latent_dim": int(best["latent_dim"]),
            "epochs": int(best["epochs"]),
            "batch_size": int(best["batch_size"]),
            "lr": float(best["lr"]),
        },
        sort_keys=True,
    )
    return df, best, model_cache[best_key]


def run_hybrid_search(
    if_model: IsolationForest,
    ae_model: TunableAutoencoder,
    ae_scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float,
    weight_grid: list[float] | None = None,
    percentile_grid: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if weight_grid is None:
        weight_grid = [0.2, 0.4, 0.6, 0.8]
    if percentile_grid is None:
        percentile_grid = [95.0, 97.0, 99.0]
    rows: list[dict] = []

    train_if = -if_model.decision_function(X_train)
    val_if = -if_model.decision_function(X_val)
    train_if_min, train_if_max = float(np.min(train_if)), float(np.max(train_if))
    den_if = max(train_if_max - train_if_min, 1e-12)
    train_if_norm = (train_if - train_if_min) / den_if
    val_if_norm = (val_if - train_if_min) / den_if

    train_ae, _ = predict_ae_scores(ae_model, ae_scaler, X_train)
    val_ae, _ = predict_ae_scores(ae_model, ae_scaler, X_val)
    train_ae_min, train_ae_max = float(np.min(train_ae)), float(np.max(train_ae))
    den_ae = max(train_ae_max - train_ae_min, 1e-12)
    train_ae_norm = (train_ae - train_ae_min) / den_ae
    val_ae_norm = (val_ae - train_ae_min) / den_ae

    for w_if in weight_grid:
        train_score = w_if * train_if_norm + (1.0 - w_if) * train_ae_norm
        val_score = w_if * val_if_norm + (1.0 - w_if) * val_ae_norm
        for perc in percentile_grid:
            thr = float(np.percentile(train_score, perc))
            preds = (val_score > thr).astype(int)
            metrics = compute_metrics_dict(y_val.to_numpy(), preds)
            rows.append(
                {
                    "weight_if": w_if,
                    "weight_ae": 1.0 - w_if,
                    "threshold_percentile": perc,
                    "threshold_value": thr,
                    **metrics,
                }
            )

    df = pd.DataFrame(rows).sort_values("attack_f1", ascending=False).reset_index(drop=True)
    best = select_best_candidate(df, min_recall=min_recall)
    return df, best


def evaluate_on_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    if_model: IsolationForest,
    ae_model: TunableAutoencoder,
    ae_scaler: StandardScaler,
    ae_threshold: float,
    hybrid_weight_if: float,
    hybrid_threshold: float,
) -> pd.DataFrame:
    if_pred = np.where(if_model.predict(X_test) == 1, 0, 1)
    if_metrics = compute_metrics_dict(y_test.to_numpy(), if_pred)

    ae_scores, ae_infer = predict_ae_scores(ae_model, ae_scaler, X_test)
    ae_pred = (ae_scores > ae_threshold).astype(int)
    ae_metrics = compute_metrics_dict(y_test.to_numpy(), ae_pred)
    ae_metrics["inference_time_s"] = ae_infer

    train_if = -if_model.decision_function(X_train)
    if_score = -if_model.decision_function(X_test)
    if_min, if_max = float(train_if.min()), float(train_if.max())
    if_norm = (if_score - if_min) / max(if_max - if_min, 1e-12)
    train_ae, _ = predict_ae_scores(ae_model, ae_scaler, X_train)
    ae_min, ae_max = float(train_ae.min()), float(train_ae.max())
    ae_norm = (ae_scores - ae_min) / max(ae_max - ae_min, 1e-12)
    hybrid_score = hybrid_weight_if * if_norm + (1.0 - hybrid_weight_if) * ae_norm
    hybrid_pred = (hybrid_score > hybrid_threshold).astype(int)
    hybrid_metrics = compute_metrics_dict(y_test.to_numpy(), hybrid_pred)

    rows = {
        "Isolation Forest tuned": if_metrics,
        "Autoencoder tuned": ae_metrics,
        "Hybrid secondary": hybrid_metrics,
    }
    return pd.DataFrame.from_dict(rows, orient="index").reset_index(names="model")


def run_experiment(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    set_global_seed(cfg.get("seed"))

    data_path = Path(cfg.get("data_path", "data/cicids2017_cleaned.csv"))
    row_limit = cfg.get("row_limit")
    seed = cfg.get("seed")
    min_recall = float(cfg.get("min_recall", 0.90))
    if_grid = cfg.get("if_grid")
    ae_grid = cfg.get("ae_grid")
    ae_percentile_grid = cfg.get("ae_threshold_percentiles")
    hybrid_weight_grid = cfg.get("hybrid_weight_grid")
    hybrid_percentile_grid = cfg.get("hybrid_threshold_percentiles")
    out_dir = Path(cfg.get("output_dir", "reports/tables"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = add_binary_label(load_data(data_path, row_limit=row_limit))
    X_train, X_val, y_val, X_test, y_test = split_protocol(df, seed=seed)

    if_df, if_best = run_if_search(
        X_train,
        X_val,
        y_val,
        min_recall=min_recall,
        seed=seed,
        param_grid=if_grid,
    )
    if_df.to_csv(out_dir / "experiment_if_candidates.csv", index=False)

    if_model = IsolationForest(
        n_estimators=int(if_best["n_estimators"]),
        contamination=float(if_best["contamination"]),
        max_samples=(
            if_best["max_samples"]
            if isinstance(if_best["max_samples"], str)
            else int(if_best["max_samples"])
        ),
        random_state=seed,
        n_jobs=-1,
    )
    if_model.fit(X_train)

    ae_df, ae_best, ae_artifacts = run_ae_search(
        X_train,
        X_val,
        y_val,
        min_recall=min_recall,
        seed=seed,
        ae_grid=ae_grid,
        percentile_grid=ae_percentile_grid,
    )
    ae_df.to_csv(out_dir / "experiment_ae_candidates.csv", index=False)

    hybrid_df, hybrid_best = run_hybrid_search(
        if_model=if_model,
        ae_model=ae_artifacts["model"],
        ae_scaler=ae_artifacts["scaler"],
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
        min_recall=min_recall,
        weight_grid=hybrid_weight_grid,
        percentile_grid=hybrid_percentile_grid,
    )
    hybrid_df.to_csv(out_dir / "experiment_hybrid_candidates.csv", index=False)

    comparison = evaluate_on_test(
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        if_model=if_model,
        ae_model=ae_artifacts["model"],
        ae_scaler=ae_artifacts["scaler"],
        ae_threshold=float(ae_best["threshold_value"]),
        hybrid_weight_if=float(hybrid_best["weight_if"]),
        hybrid_threshold=float(hybrid_best["threshold_value"]),
    )
    comparison.to_csv(out_dir / "experiment_model_comparison_test.csv", index=False)

    selection_notes = [
        "# Experiment Selection Report",
        "",
        "## Protocol",
        "- train_normal / val_mixed / test_mixed split",
        "- label mapping: normal=0, attack=1",
        f"- min_recall rule: {min_recall}",
        "",
        "## Best Validation Candidates",
        f"- IF: {if_best.to_dict()}",
        f"- AE: {ae_best.to_dict()}",
        f"- Hybrid: {hybrid_best.to_dict()}",
        "",
        "## Test Comparison File",
        f"- `{out_dir / 'experiment_model_comparison_test.csv'}`",
    ]
    (out_dir / "experiment_selection_report.md").write_text("\n".join(selection_notes), encoding="utf-8")
    print("Experiment runner completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fair model selection experiment")
    parser.add_argument("--config-path", type=Path, default=Path("config/model_selection_experiment.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(config_path=args.config_path)
