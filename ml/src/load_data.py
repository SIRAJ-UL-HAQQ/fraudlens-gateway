import os
import numpy as np
import pandas as pd

def generate_synthetic_fraud_data(n_samples: int = 5000, fraud_ratio: float = 0.02, random_state: int = 42) -> pd.DataFrame:
    """
    Generates synthetic transaction records mimicking standard Kaggle credit card PCA data.
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legit samples: low amounts, normal PCA distributions
    legit_features = np.random.normal(0, 1, size=(n_legit, 10))
    legit_amounts = np.random.exponential(scale=65.0, size=(n_legit, 1))
    legit_labels = np.zeros((n_legit, 1))

    # Fraud samples: higher amounts, shifted distributions
    fraud_features = np.random.normal(1.5, 1.8, size=(n_fraud, 10))
    fraud_amounts = np.random.exponential(scale=350.0, size=(n_fraud, 1))
    fraud_labels = np.ones((n_fraud, 1))

    legit_data = np.hstack([legit_features, legit_amounts, legit_labels])
    fraud_data = np.hstack([fraud_features, fraud_amounts, fraud_labels])

    all_data = np.vstack([legit_data, fraud_data])
    np.random.shuffle(all_data)

    cols = [f"V{i}" for i in range(1, 11)] + ["Amount", "Class"]
    df = pd.DataFrame(all_data, columns=cols)
    df["Class"] = df["Class"].astype(int)
    return df

def load_dataset(raw_path: str = "ml/data/raw/creditcard.csv") -> pd.DataFrame:
    """
    Loads raw CSV if present, or returns synthetic benchmark data if absent.
    """
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    print(f"Notice: '{raw_path}' not found. Generating reproducible synthetic benchmark dataset.")
    return generate_synthetic_fraud_data()
