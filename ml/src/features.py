"""PaySim feature engineering and preprocessor pipeline module.

Source of Truth: docs/ml-artifact-contract.md and docs/api-contract.md.
Provisional Module: Subject to extension when EDA findings from the data engineering
team arrive, as documented in MODEL_CARD.md.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

KNOWN_TRANSACTION_TYPES: list[str] = [
    "TRANSFER",
    "CASH_OUT",
    "PAYMENT",
    "CASH_IN",
    "DEBIT",
]

API_INPUT_COLUMNS: list[str] = [
    "step",
    "type",
    "amount",
    "name_orig",
    "old_balance_orig",
    "new_balance_orig",
    "name_dest",
    "old_balance_dest",
    "new_balance_dest",
]


class PaySimFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Self-contained scikit-learn transformer for PaySim transactions.

    Accepts DataFrame conforming to API snake_case schema (9 columns):
    - Drops raw `step` and extracts cyclical `hour_of_day = step % 24`.
    - Drops raw non-generalizing account IDs (`name_orig`, `name_dest`).
    - One-hot encodes `type` into KNOWN_TRANSACTION_TYPES. Unknown types produce all zeros.
    - When `include_balance_errors=True` (Variant A): calculates balance error and delta features.
    - When `include_balance_errors=False` (Variant B): omits balance error and delta features
      to prevent synthetic near-linear separability.

    Zero serving drift: exactly identical logic used offline and during backend inference.
    """

    def __init__(self, include_balance_errors: bool = False) -> None:
        self.include_balance_errors = include_balance_errors
        self.known_types_: list[str] = list(KNOWN_TRANSACTION_TYPES)
        self.input_columns_: list[str] = list(API_INPUT_COLUMNS)
        self.feature_names_in_: np.ndarray | None = None
        self.n_features_in_: int = 0

    def fit(self, X: Any, y: Any = None) -> "PaySimFeaturePreprocessor":
        """Fit preprocessor to training data, recording input schema attributes."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(list(X.columns), dtype=object)
            self.n_features_in_ = len(X.columns)
        else:
            self.feature_names_in_ = np.array(self.input_columns_, dtype=object)
            self.n_features_in_ = len(self.input_columns_)
        return self

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names for downstream estimators and SHAP explainers."""
        base_features = [
            "amount",
            "old_balance_orig",
            "new_balance_orig",
            "old_balance_dest",
            "new_balance_dest",
            "hour_of_day",
        ]

        if self.include_balance_errors:
            error_features = [
                "balance_error_orig",
                "balance_error_dest",
                "balance_delta_orig",
                "balance_delta_dest",
            ]
            type_features = [f"type_{t}" for t in self.known_types_]
            return np.array(base_features + error_features + type_features, dtype=object)

        type_features = [f"type_{t}" for t in self.known_types_]
        return np.array(base_features + type_features, dtype=object)

    def transform(self, X: Any) -> np.ndarray:
        """Transform input transaction records into a numeric feature matrix."""
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, dict):
                df = pd.DataFrame([X])
            elif isinstance(X, (list, tuple)) and X and isinstance(X[0], dict):
                df = pd.DataFrame(X)
            else:
                df = pd.DataFrame(X, columns=self.input_columns_)
        else:
            df = X

        n_samples = len(df)
        if n_samples == 0:
            names = self.get_feature_names_out()
            return np.empty((0, len(names)), dtype=np.float64)

        # 1. Base numeric columns
        amount = df["amount"].to_numpy(dtype=np.float64, copy=False)
        old_orig = df["old_balance_orig"].to_numpy(dtype=np.float64, copy=False)
        new_orig = df["new_balance_orig"].to_numpy(dtype=np.float64, copy=False)
        old_dest = df["old_balance_dest"].to_numpy(dtype=np.float64, copy=False)
        new_dest = df["new_balance_dest"].to_numpy(dtype=np.float64, copy=False)

        # 2. Cyclical hour of day derived from step
        step_arr = df["step"].to_numpy(dtype=np.int64, copy=False)
        hour_of_day = (step_arr % 24).astype(np.float64)

        arrays_to_stack = [
            amount,
            old_orig,
            new_orig,
            old_dest,
            new_dest,
            hour_of_day,
        ]

        # 3. Optional balance error and delta features (Variant A)
        if self.include_balance_errors:
            balance_error_orig = old_orig - amount - new_orig
            balance_error_dest = old_dest + amount - new_dest
            balance_delta_orig = new_orig - old_orig
            balance_delta_dest = new_dest - old_dest
            arrays_to_stack.extend([
                balance_error_orig,
                balance_error_dest,
                balance_delta_orig,
                balance_delta_dest,
            ])

        # 4. One-hot encoding of transaction type
        type_series = df["type"].astype(str)
        for t in self.known_types_:
            is_type = (type_series == t).to_numpy(dtype=np.float64)
            arrays_to_stack.append(is_type)

        matrix = np.column_stack(arrays_to_stack)
        return matrix
