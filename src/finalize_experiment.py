"""
Generate frozen final benchmark artifacts from a single config file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_models import run_benchmark_all, run_hybrid_weight_sweep


def run_final_package(config_path: Path) -> None:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    data_path = Path(cfg.get("data_path", "data/cicids2017_cleaned.csv"))
    row_limit = cfg.get("row_limit", 250000)
    contamination = float(cfg.get("contamination", 0.36))
    ae_epochs = int(cfg.get("ae_epochs", 20))
    seed_value = cfg.get("seed", 42)
    seed = int(seed_value) if seed_value is not None else None
    run_mode = cfg.get("run_mode", "")
    ae_fixed_threshold = cfg.get("ae_fixed_threshold")
    hybrid_fixed_threshold = cfg.get("hybrid_fixed_threshold")
    max_fpr = float(cfg.get("max_fpr", 0.35))
    weight_if = float(cfg.get("weight_if", 0.5))
    weight_grid = tuple(float(x) for x in cfg.get("weight_grid", [0.2, 0.4, 0.6, 0.8]))
    outputs = cfg.get("outputs", {})
    benchmark_all_path = Path(outputs.get("benchmark_all_path", "reports/tables/model_benchmark_all.csv"))
    weight_sweep_path = Path(outputs.get("weight_sweep_path", "reports/tables/hybrid_weight_sweep.csv"))
    summary_path = Path(outputs.get("summary_path", "reports/final_summary.md"))
    lock_path = Path(outputs.get("lock_path", "reports/final_locked_params.json"))

    benchmark_result = run_benchmark_all(
        data_path=data_path,
        row_limit=row_limit,
        contamination=contamination,
        ae_epochs=ae_epochs,
        weight_if=weight_if,
        seed=seed,
        ae_fixed_threshold=ae_fixed_threshold,
        hybrid_fixed_threshold=hybrid_fixed_threshold,
        max_fpr=max_fpr,
        benchmark_output=benchmark_all_path,
    )

    sweep_df = run_hybrid_weight_sweep(
        data_path=data_path,
        row_limit=row_limit,
        contamination=contamination,
        ae_epochs=ae_epochs,
        seed=seed,
        hybrid_fixed_threshold=hybrid_fixed_threshold,
        max_fpr=max_fpr,
        weights=weight_grid,
        output_path=weight_sweep_path,
    )

    best_row = {}
    if not sweep_df.empty:
        if run_mode == "official_final_frozen":
            matching = sweep_df[np.isclose(sweep_df["weight_if"], weight_if)]
            if not matching.empty:
                best_row = matching.iloc[0].to_dict()
            else:
                best_row = sweep_df.iloc[0].to_dict()
        else:
            best_row = sweep_df.iloc[0].to_dict()
    summary_title = "Final Experiment Summary" if run_mode == "official_final_frozen" else "Experiment Run Summary"
    summary_lines = [
        f"# {summary_title}",
        "",
        "## Configuration",
        f"- data_path: `{data_path}`",
        f"- row_limit: `{row_limit}`",
        f"- contamination: `{contamination}`",
        f"- ae_epochs: `{ae_epochs}`",
        f"- seed: `{seed}`",
        f"- run_mode: `{run_mode}`",
        f"- ae_fixed_threshold: `{ae_fixed_threshold}`",
        f"- hybrid_fixed_threshold: `{hybrid_fixed_threshold}`",
        f"- default_hybrid_weight_if: `{weight_if}`",
        f"- max_fpr: `{max_fpr}`",
        f"- weight_grid: `{list(weight_grid)}`",
        "",
        "## Produced Artifacts",
        f"- benchmark_all: `{benchmark_all_path}`",
        f"- hybrid_weight_sweep: `{weight_sweep_path}`",
        "",
        "## Official Hybrid Weight",
    ]
    if best_row:
        summary_lines.extend(
            [
                f"- weight_if: `{best_row.get('weight_if')}`",
                f"- attack_f1: `{best_row.get('attack_f1')}`",
                f"- attack_recall: `{best_row.get('attack_recall')}`",
                f"- false_positive_rate: `{best_row.get('false_positive_rate')}`",
            ]
        )
    else:
        summary_lines.append("- No sweep results were produced.")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Final summary saved to {summary_path}")

    locked_payload = {
        "frozen_config": {
            "data_path": str(data_path),
            "row_limit": row_limit,
            "contamination": contamination,
            "ae_epochs": ae_epochs,
            "seed": seed,
            "ae_fixed_threshold": ae_fixed_threshold,
            "hybrid_fixed_threshold": hybrid_fixed_threshold,
            "weight_if": weight_if,
            "max_fpr": max_fpr,
            "weight_grid": list(weight_grid),
        },
        "artifacts": {
            "benchmark_all_path": str(benchmark_all_path),
            "weight_sweep_path": str(weight_sweep_path),
            "summary_path": str(summary_path),
        },
        "selected_thresholds": {
            "autoencoder_threshold": benchmark_result["autoencoder"]["threshold"],
            "hybrid_threshold": benchmark_result["hybrid"]["threshold"],
        },
        "recommended_weight_by_attack_f1": best_row,
    }
    if run_mode == "official_final_frozen":
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(locked_payload, indent=2), encoding="utf-8")
        print(f"Final locked params saved to {lock_path}")
    else:
        print("Experiment mode active: lock file was not written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and generate final experiment artifacts")
    parser.add_argument("--config-path", type=Path, default=Path("config/final_config.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_final_package(config_path=args.config_path)
