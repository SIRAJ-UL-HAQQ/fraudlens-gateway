"""Class imbalance handling strategies for PaySim fraud detection.

Strategies:
1. 'none': Undersampled 10:1 baseline, natural weighting (scale_pos_weight=1.0)
2. 'class_weight': Cost-sensitive learning using inverse frequency scale_pos_weight
3. 'smote_enn': Hybrid over-sampling (SMOTE) and cleaning (Edited Nearest Neighbors)
   implemented strictly via imbalanced-learn (imblearn.combine.SMOTEENN).
"""

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN


def apply_majority_subsampling(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    neg_to_pos_ratio: float = 10.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
    """Subsample the majority class to a fixed ratio relative to the minority class.

    Applied identically across all strategies on the training split to guarantee
    a fair and controlled research comparison.
    """
    y_arr = np.asarray(y)
    pos_indices = np.where(y_arr == 1)[0]
    neg_indices = np.where(y_arr == 0)[0]

    n_pos = len(pos_indices)
    target_neg = int(min(len(neg_indices), n_pos * neg_to_pos_ratio))

    rng = np.random.RandomState(random_state)
    sampled_neg_indices = rng.choice(neg_indices, size=target_neg, replace=False)

    all_indices = np.concatenate([pos_indices, sampled_neg_indices])
    rng.shuffle(all_indices)

    if isinstance(X, pd.DataFrame):
        X_sub = X.iloc[all_indices].reset_index(drop=True)
    else:
        X_sub = X[all_indices]

    if isinstance(y, pd.Series):
        y_sub = y.iloc[all_indices].reset_index(drop=True)
    else:
        y_sub = y_arr[all_indices]

    return X_sub, y_sub


def apply_imbalance_strategy(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    strategy: str = "class_weight",
    random_state: int = 42,
) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray, float]:
    """Apply the specified class imbalance handling strategy.

    Returns:
        Tuple of (X_resampled, y_resampled, scale_pos_weight)
    """
    y_arr = np.asarray(y_train)
    neg_count = int(np.sum(y_arr == 0))
    pos_count = int(np.sum(y_arr == 1))
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    if strategy in ("none", "no_resample", "undersampled_10_to_1"):
        return X_train, y_train, 1.0

    if strategy == "class_weight":
        return X_train, y_train, scale_pos_weight

    if strategy == "smote_enn":
        smote_enn = SMOTEENN(random_state=random_state)
        X_res, y_res = smote_enn.fit_resample(X_train, y_train)
        return X_res, y_res, 1.0

    raise ValueError(
        f"Unknown strategy '{strategy}'. Supported: 'none', 'class_weight', 'smote_enn'."
    )
