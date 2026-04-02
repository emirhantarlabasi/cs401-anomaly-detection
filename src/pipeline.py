"""
Single entry-point pipeline runner for project phases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project pipelines from one CLI entry point")
    parser.add_argument(
        "--pipeline",
        choices=[
            "baseline",
            "feature-selection",
            "optimize-if",
            "mixed-eval",
            "autoencoder",
            "benchmark",
            "hybrid",
            "benchmark-all",
            "hybrid-weight-sweep",
            "finalize",
            "experiment-select",
            "if-advanced-tune",
            "pyod-benchmark",
            "pyod-stability-check",
            "ae-hbos-fusion",
            "attack-analysis",
        ],
        default="baseline",
        help="Pipeline to run",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/cicids2017_cleaned.csv"),
        help="Path to cleaned CICIDS CSV file (used by baseline pipeline)",
    )
    parser.add_argument("--row-limit", type=int, default=None, help="Optional row cap for baseline")
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.36,
        help="Expected anomaly ratio for Isolation Forest",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("baseline_report.txt"),
        help="Output txt path for baseline report",
    )
    parser.add_argument(
        "--if-results-output-path",
        type=Path,
        default=Path("reports/tables/if_mixed_results.csv"),
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/train_normal_sample.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test_attack_sample.csv"))
    parser.add_argument("--output-train", type=Path, default=Path("data/train_optimized.csv"))
    parser.add_argument("--output-test", type=Path, default=Path("data/test_optimized.csv"))
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument(
        "--grid-output-path",
        type=Path,
        default=Path("reports/tables/if_grid_search_results.csv"),
    )
    parser.add_argument("--ae-epochs", type=int, default=20)
    parser.add_argument("--ae-batch-size", type=int, default=512)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--ae-threshold-percentile", type=float, default=95.0)
    parser.add_argument("--ae-fixed-threshold", type=float, default=None)
    parser.add_argument("--ae-tune-threshold", action="store_true")
    parser.add_argument("--ae-max-fpr", type=float, default=0.35)
    parser.add_argument("--hybrid-fixed-threshold", type=float, default=None)
    parser.add_argument(
        "--ae-threshold-scan-path",
        type=Path,
        default=Path("reports/tables/ae_threshold_scan.csv"),
    )
    parser.add_argument(
        "--ae-results-output-path",
        type=Path,
        default=Path("reports/tables/ae_mixed_results.csv"),
    )
    parser.add_argument(
        "--benchmark-output-path",
        type=Path,
        default=Path("reports/tables/model_benchmark.csv"),
    )
    parser.add_argument("--weight-if", type=float, default=0.5)
    parser.add_argument(
        "--hybrid-threshold-scan-path",
        type=Path,
        default=Path("reports/tables/hybrid_threshold_scan.csv"),
    )
    parser.add_argument(
        "--hybrid-results-output-path",
        type=Path,
        default=Path("reports/tables/hybrid_mixed_results.csv"),
    )
    parser.add_argument(
        "--weight-grid",
        type=str,
        default="0.2,0.4,0.6,0.8",
        help="Comma-separated IF weights for hybrid sweep",
    )
    parser.add_argument(
        "--hybrid-weight-sweep-output-path",
        type=Path,
        default=Path("reports/tables/hybrid_weight_sweep.csv"),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config-path", type=Path, default=Path("config/experiment_config.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pipeline == "baseline":
        from baseline_model import run_baseline

        run_baseline(
            data_path=args.data_path,
            row_limit=args.row_limit,
            contamination=args.contamination,
            random_state=args.seed,
            report_path=args.report_path,
            results_output_path=args.if_results_output_path,
        )
    elif args.pipeline == "feature-selection":
        from feature_selection import perform_feature_selection

        perform_feature_selection(
            train_path=args.train_path,
            test_path=args.test_path,
            output_train=args.output_train,
            output_test=args.output_test,
            correlation_threshold=args.corr_threshold,
        )
    elif args.pipeline == "optimize-if":
        from optimize_parameters import grid_search_isolation_forest

        grid_search_isolation_forest(
            train_path=args.output_train,
            test_path=args.output_test,
            output_path=args.grid_output_path,
        )
    elif args.pipeline == "mixed-eval":
        from test_optimized_model import test_mixed_model

        test_mixed_model(
            train_opt_path=args.output_train,
            test_opt_path=args.output_test,
            contamination=args.contamination,
        )
    elif args.pipeline == "autoencoder":
        from autoencoder_model import run_autoencoder

        run_autoencoder(
            data_path=args.data_path,
            row_limit=args.row_limit,
            epochs=args.ae_epochs,
            batch_size=args.ae_batch_size,
            lr=args.ae_lr,
            threshold_percentile=args.ae_threshold_percentile,
            fixed_threshold=args.ae_fixed_threshold,
            tune_threshold=args.ae_tune_threshold,
            max_fpr=args.ae_max_fpr,
            threshold_scan_path=args.ae_threshold_scan_path,
            results_output_path=args.ae_results_output_path,
            random_state=args.seed,
        )
    elif args.pipeline == "benchmark":
        from benchmark_models import run_benchmark

        run_benchmark(
            data_path=args.data_path,
            row_limit=args.row_limit,
            contamination=args.contamination,
            ae_epochs=args.ae_epochs,
            seed=args.seed,
            ae_fixed_threshold=args.ae_fixed_threshold,
            max_fpr=args.ae_max_fpr,
            benchmark_output=args.benchmark_output_path,
        )
    elif args.pipeline == "hybrid":
        from hybrid_model import run_hybrid

        run_hybrid(
            data_path=args.data_path,
            row_limit=args.row_limit,
            contamination=args.contamination,
            ae_epochs=args.ae_epochs,
            ae_batch_size=args.ae_batch_size,
            ae_lr=args.ae_lr,
            weight_if=args.weight_if,
            tune_threshold=args.ae_tune_threshold,
            threshold_percentile=args.ae_threshold_percentile,
            fixed_threshold=args.hybrid_fixed_threshold,
            max_fpr=args.ae_max_fpr,
            threshold_scan_path=args.hybrid_threshold_scan_path,
            results_output_path=args.hybrid_results_output_path,
            random_state=args.seed,
        )
    elif args.pipeline == "benchmark-all":
        from benchmark_models import run_benchmark_all

        run_benchmark_all(
            data_path=args.data_path,
            row_limit=args.row_limit,
            contamination=args.contamination,
            ae_epochs=args.ae_epochs,
            weight_if=args.weight_if,
            seed=args.seed,
            ae_fixed_threshold=args.ae_fixed_threshold,
            hybrid_fixed_threshold=args.hybrid_fixed_threshold,
            max_fpr=args.ae_max_fpr,
            benchmark_output=args.benchmark_output_path,
        )
    elif args.pipeline == "hybrid-weight-sweep":
        from benchmark_models import run_hybrid_weight_sweep

        weights = tuple(float(item.strip()) for item in args.weight_grid.split(",") if item.strip())
        run_hybrid_weight_sweep(
            data_path=args.data_path,
            row_limit=args.row_limit,
            contamination=args.contamination,
            ae_epochs=args.ae_epochs,
            seed=args.seed,
            hybrid_fixed_threshold=args.hybrid_fixed_threshold,
            max_fpr=args.ae_max_fpr,
            weights=weights,
            output_path=args.hybrid_weight_sweep_output_path,
        )
    elif args.pipeline == "finalize":
        from finalize_experiment import run_final_package

        run_final_package(config_path=args.config_path)
    elif args.pipeline == "experiment-select":
        from experiment_runner import run_experiment

        run_experiment(config_path=args.config_path)
    elif args.pipeline == "if-advanced-tune":
        from if_advanced_tuning import run_if_advanced_tuning

        run_if_advanced_tuning(
            data_path=args.data_path,
            row_limit=args.row_limit,
            seed=args.seed,
            min_recall=0.9,
            max_fpr=args.ae_max_fpr,
            output_dir=Path("reports/tables"),
        )
    elif args.pipeline == "pyod-benchmark":
        from pyod_benchmark import run_pyod_benchmark

        run_pyod_benchmark(config_path=args.config_path)
    elif args.pipeline == "pyod-stability-check":
        from pyod_stability_check import run_stability_check

        run_stability_check(config_path=args.config_path)
    elif args.pipeline == "ae-hbos-fusion":
        from ae_hbos_fusion import run_fusion

        run_fusion(config_path=args.config_path)


if __name__ == "__main__":
    main()
