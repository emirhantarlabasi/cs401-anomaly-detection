"""
Feature selection stage for train/test samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def perform_feature_selection(
    train_path: Path = Path("data/train_normal_sample.csv"),
    test_path: Path = Path("data/test_attack_sample.csv"),
    output_train: Path = Path("data/train_optimized.csv"),
    output_test: Path = Path("data/test_optimized.csv"),
    correlation_threshold: float = 0.95,
) -> None:
    print("Feature selection basliyor...")

    if not train_path.exists():
        raise FileNotFoundError(f"Train dosyasi bulunamadi: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dosyasi bulunamadi: {test_path}")

    print("Dosyalar okunuyor...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    initial_col_count = len(df_train.columns)
    print(f"Baslangic sutun sayisi: {initial_col_count}")

    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    constant_cols = [col for col in numeric_cols if df_train[col].std() == 0]

    if constant_cols:
        df_train.drop(columns=constant_cols, inplace=True)
        df_test.drop(columns=constant_cols, inplace=True, errors="ignore")
    print(f"Sabit sutunlar silindi: {len(constant_cols)}")

    print("Korelasyon analizi yapiliyor...")
    corr_matrix = df_train.select_dtypes(include=[np.number]).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    if to_drop:
        df_train.drop(columns=to_drop, inplace=True)
        df_test.drop(columns=to_drop, inplace=True, errors="ignore")
    print(f"Yuksek korelasyonlu sutunlar silindi: {len(to_drop)}")

    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_test.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_train, index=False)
    df_test.to_csv(output_test, index=False)

    print("-" * 30)
    print("Islem tamamlandi")
    print(f"Eski sutun sayisi: {initial_col_count}")
    print(f"Yeni sutun sayisi: {len(df_train.columns)}")
    print(f"Kaydedildi -> {output_train}")
    print(f"Kaydedildi -> {output_test}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature selection pipeline")
    parser.add_argument("--train-path", type=Path, default=Path("data/train_normal_sample.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test_attack_sample.csv"))
    parser.add_argument("--output-train", type=Path, default=Path("data/train_optimized.csv"))
    parser.add_argument("--output-test", type=Path, default=Path("data/test_optimized.csv"))
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    perform_feature_selection(
        train_path=args.train_path,
        test_path=args.test_path,
        output_train=args.output_train,
        output_test=args.output_test,
        correlation_threshold=args.corr_threshold,
    )
