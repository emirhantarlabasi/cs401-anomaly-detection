"""
Autoencoder-based anomaly detection on CICIDS cleaned dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_preprocessing import add_binary_label, load_data, prepare_train_test
from evaluate import compute_metrics_dict, evaluate_binary
from random_utils import set_global_seed


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


def run_autoencoder(
    data_path: Path,
    row_limit: int | None = None,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    threshold_percentile: float = 95.0,
    fixed_threshold: float | None = None,
    tune_threshold: bool = False,
    threshold_grid: tuple[int, ...] = (50, 60, 70, 80, 85, 90, 95, 97, 99),
    max_fpr: float = 0.35,
    threshold_scan_path: Path | None = None,
    results_output_path: Path | None = None,
    random_state: int | None = 42,
) -> dict:
    set_global_seed(random_state)
    print(f"Loading data from {data_path} (row_limit={row_limit})")
    df = load_data(data_path, row_limit=row_limit)
    df = add_binary_label(df)
    X_train, X_test, y_test, feature_cols = prepare_train_test(df)
    print(f"Using {len(feature_cols)} features -> train: {X_train.shape}, test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    loader_kwargs = {"batch_size": batch_size, "shuffle": True}
    if random_state is not None:
        gen = torch.Generator()
        gen.manual_seed(random_state)
        loader_kwargs["generator"] = gen
    loader = DataLoader(TensorDataset(train_tensor), **loader_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=X_train_scaled.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        train_recon = model(train_tensor.to(device)).cpu().numpy()
        test_recon = model(test_tensor.to(device)).cpu().numpy()

    train_errors = np.mean((train_recon - X_train_scaled) ** 2, axis=1)
    test_errors = np.mean((test_recon - X_test_scaled) ** 2, axis=1)

    threshold_results: list[dict] = []
    if fixed_threshold is not None:
        threshold_value = float(fixed_threshold)
        y_pred = (test_errors > threshold_value).astype(int)
        best_metrics = compute_metrics_dict(y_true=y_test.to_numpy(), y_pred=y_pred)
    elif tune_threshold:
        best_metrics = None
        best_threshold = None
        best_preds = None
        for percentile in threshold_grid:
            threshold = np.percentile(train_errors, percentile)
            preds = (test_errors > threshold).astype(int)
            metrics = compute_metrics_dict(y_true=y_test.to_numpy(), y_pred=preds)
            row = {"percentile": percentile, "threshold": threshold, **metrics}
            threshold_results.append(row)

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
        threshold_value = float(np.percentile(train_errors, threshold_percentile))
        y_pred = (test_errors > threshold_value).astype(int)
        best_metrics = compute_metrics_dict(y_true=y_test.to_numpy(), y_pred=y_pred)

    if threshold_scan_path and threshold_results:
        threshold_scan_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(threshold_results).to_csv(threshold_scan_path, index=False)
        print(f"Threshold scan saved to {threshold_scan_path}")

    cm, report_text = evaluate_binary(y_true=y_test, y_pred=y_pred)
    print(f"Reconstruction threshold: {threshold_value:.6f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    if results_output_path:
        results_output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = X_test.copy()
        results_df["true_label"] = y_test.to_numpy()
        results_df["pred_label"] = y_pred
        results_df["anomaly_score"] = test_errors
        results_df["attack_type"] = df.loc[X_test.index, "Attack Type"]
        results_df.to_csv(results_output_path, index_label="index")
        print(f"Autoencoder results saved to {results_output_path}")

    return {
        "metrics": best_metrics,
        "threshold": threshold_value,
        "confusion_matrix": cm,
        "report_text": report_text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoencoder anomaly detector")
    parser.add_argument("--data-path", type=Path, default=Path("data/cicids2017_cleaned.csv"))
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--max-fpr", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--threshold-scan-path",
        type=Path,
        default=Path("reports/tables/ae_threshold_scan.csv"),
    )
    parser.add_argument(
        "--results-output-path",
        type=Path,
        default=Path("reports/tables/ae_mixed_results.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_autoencoder(
        data_path=args.data_path,
        row_limit=args.row_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threshold_percentile=args.threshold_percentile,
        fixed_threshold=args.fixed_threshold,
        tune_threshold=args.tune_threshold,
        max_fpr=args.max_fpr,
        threshold_scan_path=args.threshold_scan_path,
        results_output_path=args.results_output_path,
        random_state=args.seed,
    )
