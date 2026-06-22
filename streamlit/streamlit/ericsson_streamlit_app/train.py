"""
train.py — Train all models and save pipeline.pkl
==================================================
Usage:
    python model/train.py
    python model/train.py --data path/to/data.csv --output path/to/pipeline.pkl

Run this script whenever you want to retrain with fresh data.
The output pipeline.pkl is what the Flask app loads at startup.
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_DATA   = os.path.join(os.path.dirname(__file__), "ericsson_patent_rich_dataset.csv")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "pipeline.pkl")
TARGET         = "target_patent_count_next_quarter"
TEST_SIZE      = 0.20
RANDOM_STATE   = 42

FEATURES = [
    "year", "quarter", "title_len_words", "keyword_score",
    "kw_ai_ml", "kw_5g", "kw_cloud_edge", "kw_security", "kw_iot", "kw_network",
    "patent_count_lag1", "patent_count_lag2", "patent_count_lag4",
    "patent_count_roll4_mean", "patent_count_roll8_mean",
    "patent_count_qoq", "patent_count_yoy",
]
CAT_COLS = ["patent_type", "tech_era"]


def load_and_clean(data_path: str) -> pd.DataFrame:
    print(f"\n[1/5] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Raw shape: {df.shape}")

    df["patent_date"] = pd.to_datetime(df["patent_date"])

    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"      Dropped {before - len(df)} rows with missing target")
    print(f"      Clean shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame):
    print("\n[2/5] Encoding categorical features...")
    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

    enc_cols = [c for c in df_enc.columns if any(c.startswith(cat) for cat in CAT_COLS)]
    all_features = FEATURES + enc_cols

    print(f"      Total features: {len(all_features)}")
    print(f"      Encoded columns: {enc_cols}")

    X = df_enc[all_features]
    y = df_enc[TARGET]
    return X, y, all_features


def split_and_scale(X, y):
    print(f"\n[3/5] Splitting data (80% train / 20% test, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"      Train: {X_train.shape[0]:,} rows")
    print(f"      Test : {X_test.shape[0]:,} rows")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)       # ← NEVER refit on test data!

    return X_train_sc, X_test_sc, y_train, y_test, scaler


def train_models(X_train, y_train):
    print("\n[4/5] Training models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge"            : Ridge(alpha=1.0),
        "Lasso"            : Lasso(alpha=0.1, max_iter=5000),
        "Random Forest"    : RandomForestRegressor(
                                n_estimators=100,
                                random_state=RANDOM_STATE,
                                n_jobs=-1
                             ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"      ✅ {name}")
    return models


def evaluate(models, X_test, y_test):
    print("\n[5/5] Evaluating on test set...")
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))
        results[name] = {"r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}
        print(f"      {name:20s}  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    return results


def save_pipeline(models, scaler, features, results, output_path):
    payload = {
        "models"  : models,
        "scaler"  : scaler,
        "features": features,
        "results" : results,
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n✅ Pipeline saved → {output_path}  ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Train Ericsson patent prediction models")
    parser.add_argument("--data",   default=DEFAULT_DATA,   help="Path to CSV dataset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output path for pipeline.pkl")
    args = parser.parse_args()

    print("=" * 55)
    print("  Ericsson Patent Predictor — Training Pipeline")
    print("=" * 55)

    df               = load_and_clean(args.data)
    X, y, features   = engineer_features(df)
    X_tr, X_te, y_tr, y_te, scaler = split_and_scale(X, y)
    models           = train_models(X_tr, y_tr)
    results          = evaluate(models, X_te, y_te)
    save_pipeline(models, scaler, features, results, args.output)

    print("\n" + "=" * 55)
    print("  Training complete!")
    print("  Next step: python wsgi.py  OR  docker build")
    print("=" * 55)


if __name__ == "__main__":
    main()
