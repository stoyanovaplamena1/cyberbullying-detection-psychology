from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from paths import DATA_DIR  # ensure DATA_DIR is defined in src/paths.py

RANDOM_STATE = 42  

# Force fastparquet everywhere
PARQUET_ENGINE = "fastparquet"
pd.options.io.parquet.engine = PARQUET_ENGINE  # optional but helpful

def load_cleaned_cb(filename: str = "cleaned_cyberbullying.parquet") -> pd.DataFrame:
    """
    Load the cleaned Cyberbullying dataframe from data/.
    Expects columns like: 'cyberbullying_type', 'is_bullying' or 'any_toxic',
    and the preprocessed text fields you used for features.
    """
    return pd.read_parquet(DATA_DIR / filename, engine=PARQUET_ENGINE)

def load_cleaned_tx(filename: str = "cleaned_jigsaw.parquet") -> pd.DataFrame:
    """
    Load the cleaned Jigsaw Toxic Comment dataframe from data/.
    Expects multilabel columns: ['toxic','severe_toxic','obscene','threat','insult','identity_hate'].
    """
    return pd.read_parquet(DATA_DIR / filename, engine=PARQUET_ENGINE)

# ---------- Build labels (CB) ----------

def build_cb_binary_labels(cb_df: pd.DataFrame, splits: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    """
    Build binary labels for CB (0=clean, 1=toxic).
    Priority:
      1) use 'is_bullying' if present,
      2) else 'any_toxic' if present,
      3) else derive from 'cyberbullying_type' != 'not_cyberbullying'.
    Returns dict with numpy arrays for 'train'/'dev'/'test'.
    """
    if "is_bullying" in cb_df.columns:
        lab_all = cb_df["is_bullying"].astype(int).values
    elif "any_toxic" in cb_df.columns:
        lab_all = cb_df["any_toxic"].astype(int).values
    else:
        lab_all = (cb_df["cyberbullying_type"] != "not_cyberbullying").astype(int).values

    return {
        "train": lab_all[np.asarray(splits["train"], dtype=int)],
        "dev":   lab_all[np.asarray(splits["dev"],   dtype=int)],
        "test":  lab_all[np.asarray(splits["test"],  dtype=int)],
    }

def build_cb_multiclass_labels(
    cb_df: pd.DataFrame,
    splits: Dict[str, List[int]],
    col: str = "cyberbullying_type",
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Build multiclass labels (category codes) from 'cyberbullying_type'.
    Returns (labels_dict, class_names).
    """
    y_all = cb_df[col].astype("category")
    classes = list(y_all.cat.categories)
    y_codes = y_all.cat.codes.values
    return (
        {
            "train": y_codes[np.asarray(splits["train"], dtype=int)],
            "dev":   y_codes[np.asarray(splits["dev"],   dtype=int)],
            "test":  y_codes[np.asarray(splits["test"],  dtype=int)],
        },
        classes,
    )

# ---------- Build labels (TX) ----------

def build_tx_multilabels(
    tx_df: pd.DataFrame,
    splits: Dict[str, List[int]],
    label_order: List[str],
) -> Dict[str, np.ndarray]:
    """
    Stack multilabel columns in 'label_order' into numpy arrays per split.
    Returns dict with shape (n_split, L) integer arrays.
    """
    assert all(c in tx_df.columns for c in label_order), "Missing TX label columns in dataframe."
    Y_all = tx_df[label_order].astype(int).values
    return {
        "train": Y_all[np.asarray(splits["train"], dtype=int), :],
        "dev":   Y_all[np.asarray(splits["dev"],   dtype=int), :],
        "test":  Y_all[np.asarray(splits["test"],  dtype=int), :],
    }

def tx_toxic_any_from_multilabel(Y: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Derive harmonized binary 'toxic-any' labels from TX multilabel matrices.
    toxic_any = 1 if any label is 1, else 0. Returns dict of 0/1 arrays.
    """
    return {k: (v.max(axis=1) > 0).astype(int) for k, v in Y.items()}