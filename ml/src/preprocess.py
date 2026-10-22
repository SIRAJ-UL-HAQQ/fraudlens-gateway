import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def preprocess_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Applies RobustScaler to Amount and splits data with stratification.
    """
    df_clean = df.copy()
    
    scaler = RobustScaler()
    if "Amount" in df_clean.columns:
        df_clean["scaled_amount"] = scaler.fit_transform(df_clean[["Amount"]])
        df_clean = df_clean.drop(columns=["Amount"])

    if "Time" in df_clean.columns:
        df_clean["scaled_time"] = scaler.fit_transform(df_clean[["Time"]])
        df_clean = df_clean.drop(columns=["Time"])

    X = df_clean.drop(columns=["Class"])
    y = df_clean["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler
