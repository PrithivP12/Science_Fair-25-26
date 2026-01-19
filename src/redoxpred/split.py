from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


GROUP_COL = "uniprot_id"


def train_val_test_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> Dict[str, List[int]]:
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found")

    groups = df[group_col].astype(str).values
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        raise ValueError("Too few groups for train/val/test split")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(df, groups=groups))

    train_val_groups = groups[train_val_idx]
    val_relative = val_size / (1.0 - test_size)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed)
    train_idx, val_idx = next(splitter_val.split(train_val_idx, groups=train_val_groups))

    train_indices = train_val_idx[train_idx].tolist()
    val_indices = train_val_idx[val_idx].tolist()
    test_indices = test_idx.tolist()

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def group_kfold_indices(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int,
) -> List[Tuple[List[int], List[int]]]:
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found")
    groups = df[group_col].astype(str).values
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough groups for requested GroupKFold splits")
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for train_idx, val_idx in gkf.split(df, groups=groups):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds
