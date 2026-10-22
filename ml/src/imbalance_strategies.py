from typing import Tuple
import numpy as np
import pandas as pd

def apply_imbalance_strategy(X_train: pd.DataFrame, y_train: pd.Series, strategy: str = "class_weight") -> Tuple[pd.DataFrame, pd.Series, float]:
    """
    Supports three imbalance strategies:
    1. 'no-resample' / 'none': returns original data, scale_pos_weight = 1.0
    2. 'class-weight': calculates inverse frequency scale_pos_weight for XGBoost
    3. 'smote-enn': SMOTE oversampling combined with Edited Nearest Neighbors undersampling
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    if strategy in ("no-resample", "none"):
        return X_train, y_train, 1.0

    if strategy == "class-weight":
        return X_train, y_train, scale_pos_weight

    if strategy == "smote-enn":
        try:
            from imblearn.combine import SMOTEENN
            smote_enn = SMOTEENN(random_state=42)
            X_res, y_res = smote_enn.fit_resample(X_train, y_train)
            return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res), 1.0
        except ImportError:
            print("imblearn not installed; falling back to class-weight strategy.")
            return X_train, y_train, scale_pos_weight

    raise ValueError(f"Unknown strategy '{strategy}'. Choose from 'no-resample', 'class-weight', 'smote-enn'.")
