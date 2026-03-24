"""
Data loading and baseline-ready preprocessing utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_data(path: Path, row_limit: int | None = None) -> pd.DataFrame:
    """Read the CICIDS dataset with an optional row cap."""
    read_kwargs = {"low_memory": False}
    if row_limit is not None:
        read_kwargs["nrows"] = row_limit
    return pd.read_csv(path, **read_kwargs)


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of dataframe with binary label column."""
    out = df.copy()
    out["label"] = out["Attack Type"].apply(lambda value: 0 if value == "Normal Traffic" else 1)
    return out


def prepare_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Prepare normal-only train and mixed test matrices for IF baseline."""
    feature_cols = df.columns.drop(["Attack Type", "label"])
    train_df = df[df["label"] == 0]
    test_df = df.copy()
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]
    return X_train, X_test, y_test, feature_cols.tolist()
