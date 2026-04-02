"""
Analyze and visualize detection recall rates across different attack types.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_attack_types(
    results_csv: Path,
    output_dir: Path = Path("reports/figures")
) -> None:
    print(f"Loading results from {results_csv}...")
    if not results_csv.exists():
        raise FileNotFoundError(f"File not found: {results_csv}. Please run a model pipeline first.")

    df = pd.read_csv(results_csv)
    
    if "attack_type" not in df.columns:
        raise ValueError("Column 'attack_type' is missing from the results CSV. Ensure the model saved it.")

    # Filter only actual attacks (true_label == 1)
    attacks_df = df[df["true_label"] == 1].copy()
    
    if attacks_df.empty:
        print("No attacks found in the dataset.")
        return

    # Calculate Detection Rate (Recall) per attack type
    summary = attacks_df.groupby("attack_type").agg(
        total_samples=("true_label", "count"),
        detected=("pred_label", "sum")
    ).reset_index()
    
    summary["recall_rate"] = summary["detected"] / summary["total_samples"]
    summary = summary.sort_values("recall_rate", ascending=False)
    
    # Save table to reports/tables
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_out_path = tables_dir / f"{results_csv.stem}_attack_analysis.csv"
    summary.to_csv(csv_out_path, index=False)
    
    # Visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bar plot
    ax = sns.barplot(
        x="recall_rate", 
        y="attack_type", 
        data=summary, 
        palette="viridis",
        hue="attack_type",
        legend=False
    )
    
    plt.title(f"Attack Type Detection Rate (Recall) - {results_csv.stem}", fontsize=14, pad=15)
    plt.xlabel("Detection Rate (Recall)", fontsize=12)
    plt.ylabel("Attack Type", fontsize=12)
    plt.xlim(0, 1.05)
    
    # Add percentage labels to the bars
    for i, p in enumerate(summary["recall_rate"]):
        ax.text(p + 0.01, i, f"{p:.1%}", va="center", fontsize=10)
        
    plt.tight_layout()
    img_out_path = output_dir / f"{results_csv.stem}_attack_analysis.png"
    plt.savefig(img_out_path, dpi=150)
    plt.close()
    
    print("-" * 50)
    print("ATTACK TYPE RECALL SUMMARY:")
    print(summary.to_string(index=False))
    print("-" * 50)
    print(f"Saved analysis table: {csv_out_path}")
    print(f"Saved visualization: {img_out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze recall per attack type.")
    parser.add_argument(
        "--results-csv", 
        type=Path, 
        default=Path("reports/tables/if_mixed_results.csv"),
        help="Path to the model results CSV file"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("reports/figures"),
        help="Directory to save the visualization"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_attack_types(results_csv=args.results_csv, output_dir=args.output_dir)
