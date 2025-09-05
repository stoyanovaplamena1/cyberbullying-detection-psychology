# src/splits.py
from typing import Dict, List
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from paths import RANDOM_STATE, artifact

Splits = Dict[str, List[int]]  # {"train":[...], "dev":[...], "test":[...]}

def make_cb_splits(
    df: pd.DataFrame,
    label_col: str = "cyberbullying_type",
    test_size: float = 0.20,
    dev_frac_of_trdev: float = 0.125,
    random_state: int = RANDOM_STATE,
) -> Splits:
    """
    Create CB (Twitter-like) splits stratified by the multiclass column.
    - First split: (train+dev) vs test with stratify=label_col.
    - Second split: dev from (train+dev), stratified again.
    """
    all_idx = np.arange(len(df))
    labels = df[label_col]
    trdev_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, stratify=labels, random_state=random_state
    )
    trdev_labels = labels.iloc[trdev_idx]
    train_idx, dev_idx = train_test_split(
        trdev_idx, test_size=dev_frac_of_trdev, stratify=trdev_labels, random_state=random_state
    )
    return {
        "train": np.asarray(train_idx, dtype=int).tolist(),
        "dev":   np.asarray(dev_idx,   dtype=int).tolist(),
        "test":  np.asarray(test_idx,  dtype=int).tolist(),
    }

def make_tx_splits(
    df: pd.DataFrame,
    label_cols: List[str],
    test_size: float = 0.20,
    dev_frac_of_trdev: float = 0.125,
    random_state: int = RANDOM_STATE,
) -> Splits:
    """
    Create TX (Jigsaw) splits stratified by a coarse label count:
      bins = clip(sum(labels), 0, 2), i.e., 0, 1, or 2+.
    This matches the notebook logic and is robust to imbalance.
    """
    all_idx = np.arange(len(df))
    label_count = df[label_cols].astype(int).sum(axis=1)
    bins = label_count.clip(0, 2)

    trdev_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, stratify=bins, random_state=random_state
    )
    trdev_bins = bins.iloc[trdev_idx]
    train_idx, dev_idx = train_test_split(
        trdev_idx, test_size=dev_frac_of_trdev, stratify=trdev_bins, random_state=random_state
    )
    return {
        "train": np.asarray(train_idx, dtype=int).tolist(),
        "dev":   np.asarray(dev_idx,   dtype=int).tolist(),
        "test":  np.asarray(test_idx,  dtype=int).tolist(),
    }

def save_splits(splits: Splits, name: str) -> str:
    """
    Save splits to artifacts as `<name>_splits.json`. Returns the file path (str).
    """
    path = artifact(f"{name}_splits.json")
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    return str(path)

def load_splits(name: str) -> Splits:
    """
    Load splits from artifacts `<name>_splits.json`.
    """
    path = artifact(f"{name}_splits.json")
    with open(path, "r") as f:
        return json.load(f)

def slice_df(df: pd.DataFrame, splits: Splits) -> Dict[str, pd.DataFrame]:
    """
    Convenience: return train/dev/test DataFrames using the provided index lists.
    """
    return {
        "train": df.iloc[splits["train"]],
        "dev":   df.iloc[splits["dev"]],
        "test":  df.iloc[splits["test"]],
    }
