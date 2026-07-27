from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "open_orders_train.csv"
MODEL_PATH = ROOT / "models" / "delay_model.pkl"


def main():
    print(f"📦 Loading training data from: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Target column from generate_data.py
    target_col = "late_flag"

    # Features = everything except the target and non-generalisable columns.
    # Date strings and order_id are dropped to keep the training feature set
    # consistent with what api.py and inference.py can provide at scoring time.
    # Time-based signals are already captured by weekday_ordered and month_ordered.
    DROP_COLS = ["late_flag", "order_id", "order_date", "requested_ship_date", "promised_ship_date"]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[target_col]

    print("One-hot encoding categorical features...")
    X = pd.get_dummies(X, drop_first=True)

    print("🔀 Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🌲 Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("📊 Evaluation on test set:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "columns": X.columns.tolist()}, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    print("🚀 Starting training script...")
    main()
    print("✅ Training complete.")
