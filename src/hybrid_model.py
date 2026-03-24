"""
Hybrid anomaly detector combining Isolation Forest and Autoencoder scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from autoencoder_model import Autoencoder
from data_preprocessing import add_binary_label, load_data, prepare_train_test
from evaluate import compute_metrics_dict, evaluate_binary
from random_utils import set_global_seed


def _minmax_scale(train_values: np.ndarray, test_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_min = float(np.min(train_values))
    train_max = float(np.max(train_values))
    denom = train_max - train_min
    if denom <= 1e-12:
        return np.zeros_like(train_values), np.zeros_like(test_values)
    train_scaled = (train_values - train_min) / denom
    test_scaled = (test_values - train_min) / denom
    return train_scaled, test_scaled


def run_hybrid(
    data_path: Path,
    row_limit: int | None = None,
    contamination: float = 0.36,
    ae_epochs: int = 20,
    ae_batch_size: int = 512,
    ae_lr: float = 1e-3,
    weight_if: float = 0.5,
    tune_threshold: bool = True,
    threshold_percentile: float = 95.0,
    fixed_threshold: float | None = None,
    threshold_grid: tuple[int, ...] = (50, 60, 70, 80, 85, 90, 95, 97, 99),
    max_fpr: float = 0.35,
    threshold_scan_path: Path | None = Path("reports/tables/hybrid_threshold_scan.csv"),
    results_output_path: Path | None = None,
    random_state: int | None = 42,
) -> dict:
    set_global_seed(random_state)
    print(f"Loading data from {data_path} (row_limit={row_limit})")
    df = load_data(data_path, row_limit=row_limit)
    df = add_binary_label(df)
    X_train, X_test, y_test, feature_cols = prepare_train_test(df)
    print(f"Using {len(feature_cols)} features -> train: {X_train.shape}, test: {X_test.shape}")

    # Isolation Forest score branch
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    iso.fit(X_train_np)
    train_if_score = -iso.decision_function(X_train_np)
    test_if_score = -iso.decision_function(X_test_np)
    train_if_scaled, test_if_scaled = _minmax_scale(train_if_score, test_if_score)

    # Autoencoder score branch
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled = scaler.transform(X_test_np)
    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    loader_kwargs = {"batch_size": ae_batch_size, "shuffle": True}
    if random_state is not None:
        gen = torch.Generator()
        gen.manual_seed(random_state)
        loader_kwargs["generator"] = gen
    loader = DataLoader(TensorDataset(train_tensor), **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = Autoencoder(input_dim=X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=ae_lr)

    ae.train()
    for epoch in range(ae_epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = ae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(loader.dataset)
        print(f"AE Epoch {epoch + 1}/{ae_epochs} - loss: {epoch_loss:.6f}")

    ae.eval()
    with torch.no_grad():
        train_recon = ae(train_tensor.to(device)).cpu().numpy()
        test_recon = ae(test_tensor.to(device)).cpu().numpy()
    train_ae_score = np.mean((train_recon - X_train_scaled) ** 2, axis=1)
    test_ae_score = np.mean((test_recon - X_test_scaled) ** 2, axis=1)
    train_ae_scaled, test_ae_scaled = _minmax_scale(train_ae_score, test_ae_score)

    # Hybrid score
    w_if = float(np.clip(weight_if, 0.0, 1.0))
    w_ae = 1.0 - w_if
    train_hybrid = w_if * train_if_scaled + w_ae * train_ae_scaled
    test_hybrid = w_if * test_if_scaled + w_ae * test_ae_scaled

    threshold_results: list[dict] = []
    if fixed_threshold is not None:
        threshold_value = float(fixed_threshold)
        y_pred = (test_hybrid > threshold_value).astype(int)
        best_metrics = compute_metrics_dict(y_true=y_test.to_numpy(), y_pred=y_pred)
    elif tune_threshold:
        best_metrics = None
        best_threshold = None
        best_preds = None
        for percentile in threshold_grid:
            threshold = np.percentile(train_hybrid, percentile)
            preds = (test_hybrid > threshold).astype(int)
            metrics = compute_metrics_dict(y_true=y_test.to_numpy(), y_pred=preds)
            threshold_results.append({"percentile": percentile, "threshold": threshold, **metrics})

            allowed = metrics["false_positive_rate"] <= max_fpr
            if best_metrics is None:
                best_metrics = metrics
                best_threshold = threshold
                best_preds = preds
            else:
                best_allowed = best_metrics["false_positive_rate"] <= max_fpr
                replace = False
                if allowed and not best_allowed:
                    replace = True
                elif allowed == best_allowed and metrics["attack_f1"] > best_metrics["attack_f1"]:
                    replace = True
                if replace:
                    best_metrics = metrics
                    best_threshold = threshold
                    best_preds = preds
        threshold_value = float(best_threshold)
        y_pred = best_preds
    else:
        threshold_value = float(np.percentile(train_hybrid, threshold_percentile))
        y_pred = (test_hybrid > threshold_value).astype(int)
        best_metrics = compute_metrics_dict(y_true=y_test.to_numpy(), y_pred=y_pred)

    if threshold_scan_path and threshold_results:
        threshold_scan_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(threshold_results).to_csv(threshold_scan_path, index=False)
        print(f"Hybrid threshold scan saved to {threshold_scan_path}")

    cm, report_text = evaluate_binary(y_true=y_test, y_pred=y_pred)
    print(f"Hybrid threshold: {threshold_value:.6f} (w_if={w_if:.2f}, w_ae={w_ae:.2f})")
    print("Confusion matrix:")
    print(cm)

    if results_output_path:
        results_output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = X_test.copy()
        results_df["true_label"] = y_test.to_numpy()
        results_df["pred_label"] = y_pred
        results_df["if_score"] = test_if_scaled
        results_df["ae_score"] = test_ae_scaled
        results_df["anomaly_score"] = test_hybrid
        results_df.to_csv(results_output_path, index_label="index")
        print(f"Hybrid results saved to {results_output_path}")

    return {
        "metrics": best_metrics,
        "threshold": threshold_value,
        "confusion_matrix": cm,
        "report_text": report_text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid IF + Autoencoder anomaly detector")
    parser.add_argument("--data-path", type=Path, default=Path("data/cicids2017_cleaned.csv"))
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--contamination", type=float, default=0.36)
    parser.add_argument("--ae-epochs", type=int, default=20)
    parser.add_argument("--ae-batch-size", type=int, default=512)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--weight-if", type=float, default=0.5)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--max-fpr", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--threshold-scan-path",
        type=Path,
        default=Path("reports/tables/hybrid_threshold_scan.csv"),
    )
    parser.add_argument(
        "--results-output-path",
        type=Path,
        default=Path("reports/tables/hybrid_mixed_results.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hybrid(
        data_path=args.data_path,
        row_limit=args.row_limit,
        contamination=args.contamination,
        ae_epochs=args.ae_epochs,
        ae_batch_size=args.ae_batch_size,
        ae_lr=args.ae_lr,
        weight_if=args.weight_if,
        tune_threshold=args.tune_threshold,
        threshold_percentile=args.threshold_percentile,
        fixed_threshold=args.fixed_threshold,
        max_fpr=args.max_fpr,
        threshold_scan_path=args.threshold_scan_path,
        results_output_path=args.results_output_path,
        random_state=args.seed,
    )
