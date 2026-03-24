"""Benchmark runners for IF, Autoencoder and Hybrid models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from autoencoder_model import run_autoencoder
from baseline_model import run_baseline
from evaluate import save_benchmark_table
from hybrid_model import run_hybrid


def run_benchmark(
    data_path: Path,
    row_limit: int | None = None,
    contamination: float = 0.36,
    ae_epochs: int = 20,
    seed: int | None = 42,
    ae_fixed_threshold: float | None = None,
    max_fpr: float = 0.35,
    benchmark_output: Path = Path("reports/tables/model_benchmark.csv"),
) -> dict:
    print("Running Isolation Forest baseline...")
    baseline_result = run_baseline(
        data_path=data_path,
        row_limit=row_limit,
        contamination=contamination,
        random_state=seed,
        report_path=None,
        results_output_path=Path("reports/tables/if_mixed_results.csv"),
    )
    print("\nRunning Autoencoder model...")
    ae_result = run_autoencoder(
        data_path=data_path,
        row_limit=row_limit,
        epochs=ae_epochs,
        fixed_threshold=ae_fixed_threshold,
        tune_threshold=ae_fixed_threshold is None,
        max_fpr=max_fpr,
        threshold_scan_path=Path("reports/tables/ae_threshold_scan.csv"),
        results_output_path=Path("reports/tables/ae_mixed_results.csv"),
        random_state=seed,
    )

    rows = {
        "Isolation Forest": baseline_result["metrics"],
        "Autoencoder": ae_result["metrics"],
    }
    df = save_benchmark_table(rows=rows, output_path=benchmark_output)
    print(f"\nBenchmark table saved to {benchmark_output}")
    print(df.to_string(index=False))
    return {"table": df, "baseline": baseline_result, "autoencoder": ae_result}


def run_benchmark_all(
    data_path: Path,
    row_limit: int | None = None,
    contamination: float = 0.36,
    ae_epochs: int = 20,
    weight_if: float = 0.5,
    seed: int | None = 42,
    ae_fixed_threshold: float | None = None,
    hybrid_fixed_threshold: float | None = None,
    max_fpr: float = 0.35,
    benchmark_output: Path = Path("reports/tables/model_benchmark_all.csv"),
    manifest_output: Path = Path("reports/tables/evaluation_manifest.json"),
) -> dict:
    print("Running Isolation Forest baseline...")
    baseline_result = run_baseline(
        data_path=data_path,
        row_limit=row_limit,
        contamination=contamination,
        random_state=seed,
        report_path=None,
        results_output_path=Path("reports/tables/if_mixed_results.csv"),
    )

    print("\nRunning Autoencoder model...")
    ae_result = run_autoencoder(
        data_path=data_path,
        row_limit=row_limit,
        epochs=ae_epochs,
        fixed_threshold=ae_fixed_threshold,
        tune_threshold=ae_fixed_threshold is None,
        max_fpr=max_fpr,
        threshold_scan_path=Path("reports/tables/ae_threshold_scan.csv"),
        results_output_path=Path("reports/tables/ae_mixed_results.csv"),
        random_state=seed,
    )

    print("\nRunning Hybrid model...")
    hybrid_result = run_hybrid(
        data_path=data_path,
        row_limit=row_limit,
        contamination=contamination,
        ae_epochs=ae_epochs,
        weight_if=weight_if,
        fixed_threshold=hybrid_fixed_threshold,
        tune_threshold=hybrid_fixed_threshold is None,
        max_fpr=max_fpr,
        threshold_scan_path=Path("reports/tables/hybrid_threshold_scan.csv"),
        results_output_path=Path("reports/tables/hybrid_mixed_results.csv"),
        random_state=seed,
    )

    rows = {
        "Isolation Forest": baseline_result["metrics"],
        "Autoencoder": ae_result["metrics"],
        "Hybrid (IF+AE)": hybrid_result["metrics"],
    }
    df = save_benchmark_table(rows=rows, output_path=benchmark_output)
    print(f"\nFull benchmark table saved to {benchmark_output}")
    print(df.to_string(index=False))
    manifest = {
        "data_path": str(data_path),
        "row_limit": row_limit,
        "label_mapping": {"normal": 0, "attack": 1},
        "train_policy": "train on normal-only subset",
        "test_policy": "evaluate on full mixed dataset",
        "seed": seed,
        "contamination": contamination,
        "weight_if": weight_if,
        "ae_fixed_threshold": ae_fixed_threshold,
        "hybrid_fixed_threshold": hybrid_fixed_threshold,
        "max_fpr": max_fpr,
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Evaluation manifest saved to {manifest_output}")
    return {"table": df, "baseline": baseline_result, "autoencoder": ae_result, "hybrid": hybrid_result}


def run_hybrid_weight_sweep(
    data_path: Path,
    row_limit: int | None = None,
    contamination: float = 0.36,
    ae_epochs: int = 20,
    seed: int | None = 42,
    hybrid_fixed_threshold: float | None = None,
    max_fpr: float = 0.35,
    weights: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
    output_path: Path = Path("reports/tables/hybrid_weight_sweep.csv"),
) -> pd.DataFrame:
    print("Running hybrid weight sweep...")
    rows: list[dict] = []

    for weight_if in weights:
        print(f"\nWeight test -> weight_if={weight_if:.2f}, weight_ae={1.0 - weight_if:.2f}")
        result = run_hybrid(
            data_path=data_path,
            row_limit=row_limit,
            contamination=contamination,
            ae_epochs=ae_epochs,
            weight_if=weight_if,
            fixed_threshold=hybrid_fixed_threshold,
            tune_threshold=hybrid_fixed_threshold is None,
            max_fpr=max_fpr,
            threshold_scan_path=None,
            random_state=seed,
        )
        rows.append(
            {
                "weight_if": weight_if,
                "weight_ae": 1.0 - weight_if,
                "selected_threshold": result["threshold"],
                **result["metrics"],
            }
        )

    df = pd.DataFrame(rows).sort_values("attack_f1", ascending=False).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nHybrid weight sweep saved to {output_path}")
    print(df.to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark IF vs Autoencoder")
    parser.add_argument("--data-path", type=Path, default=Path("data/cicids2017_cleaned.csv"))
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--contamination", type=float, default=0.36)
    parser.add_argument("--ae-epochs", type=int, default=20)
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("reports/tables/model_benchmark.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        data_path=args.data_path,
        row_limit=args.row_limit,
        contamination=args.contamination,
        ae_epochs=args.ae_epochs,
        benchmark_output=args.benchmark_output,
    )
