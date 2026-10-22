import os
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from load_data import load_dataset
from preprocess import preprocess_and_split
from imbalance_strategies import apply_imbalance_strategy
from evaluate import evaluate_predictions

def train_pipeline(strategy: str = "class-weight", output_dir: str = "ml/models"):
    print(f"--- Starting FraudLens Model Training (Strategy: {strategy}) ---")
    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df)

    X_train_res, y_train_res, scale_pos_weight = apply_imbalance_strategy(X_train, y_train, strategy=strategy)

    # 1. Baseline: Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_probs)
    print(f"Baseline Logistic Regression PR-AUC: {lr_metrics['pr_auc']}, F1: {lr_metrics['f1']}")

    # 2. Main Candidate: XGBoost
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42
    )
    xgb.fit(X_train_res, y_train_res)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_predictions(y_test, xgb_probs)
    print(f"XGBoost Classifier PR-AUC: {xgb_metrics['pr_auc']}, F1: {xgb_metrics['f1']}, MCC: {xgb_metrics['mcc']}")

    # Persist artifact
    os.makedirs(output_dir, exist_ok=True)
    model_artifact = {
        "model": xgb,
        "scaler": scaler,
        "metrics": xgb_metrics,
        "strategy": strategy,
        "features": list(X_train.columns)
    }
    model_path = os.path.join(output_dir, "xgboost_fraud_model.joblib")
    joblib.dump(model_artifact, model_path)
    print(f"Model artifact saved successfully to: {model_path}")

if __name__ == "__main__":
    train_pipeline()
