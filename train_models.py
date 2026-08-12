"""
train_models.py
===============
AI-Powered Network Traffic Analyzer — TRAINING PHASE

Reads NF-UNSW-NB15-v3.csv, replicates the exact training pipeline from the
original notebook, and saves the following artifacts to models/:

  models/rf_model.pkl       — trained RandomForestClassifier (unscaled input)
  models/ae_model.keras     — trained Autoencoder (scaled input)
  models/scaler.pkl         — fitted StandardScaler
  models/artifacts.json     — feature_cols, ae_threshold, ae_mse_max_train

Feature schema is imported from feature_schema.py — do NOT define FEATURE_COLS here.

Run once:
    python train_models.py

After this script completes, start the live dashboard:
    python app.py
"""

import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "NF-UNSW-NB15-v3.csv")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")

os.makedirs(MODELS_DIR, exist_ok=True)

RF_PATH        = os.path.join(MODELS_DIR, "rf_model.pkl")
AE_PATH        = os.path.join(MODELS_DIR, "ae_model.keras")
SCALER_PATH    = os.path.join(MODELS_DIR, "scaler.pkl")
ARTIFACTS_PATH = os.path.join(MODELS_DIR, "artifacts.json")

# ── Feature schema — imported from the single canonical source ───────────────
from feature_schema import FEATURE_COLS  # noqa: E402


# ============================================================
# 1. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place NF-UNSW-NB15-v3.csv in the same folder as this script."
    )

print("Loading dataset …")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape}")


# ============================================================
# 2. CREATE BINARY TARGET  (Normal = 0, Attack = 1)
# ============================================================

df["label_enc"] = np.where(
    df["Attack"].astype(str).str.strip().str.lower() == "benign",
    0,
    1,
).astype("int8")

print("\nClass distribution:")
print(df["label_enc"].value_counts())


# ============================================================
# 3. FEATURE MATRIX
# ============================================================

X = df[FEATURE_COLS].copy()

for col in FEATURE_COLS:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
y = df["label_enc"]

print(f"\nFeature matrix : {X.shape}")
print(f"Missing values : {X.isna().sum().sum()}")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")


# ============================================================
# 5. STANDARD SCALER  (fit on train, transform test)
# ============================================================

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)

print("\nScaler fitted.")


# ============================================================
# 6. RANDOM FOREST  (trained on UNSCALED features — matches notebook)
# ============================================================

print("\nTraining Random Forest …")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
)
rf.fit(X_train, y_train)           # NOTE: unscaled input

y_pred       = rf.predict(X_test)
attack_idx   = np.where(rf.classes_ == 1)[0][0]
rf_probs     = rf.predict_proba(X_test)[:, attack_idx]

rf_accuracy  = accuracy_score(y_test, y_pred)
rf_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rf_recall    = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
rf_f1        = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

print("\nRandom Forest Results")
print("--------------------------------")
print(f"Accuracy : {rf_accuracy:.6f}")
print(f"Precision: {rf_precision:.6f}")
print(f"Recall   : {rf_recall:.6f}")
print(f"F1 Score : {rf_f1:.6f}")
print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred,
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# 7. AUTOENCODER  (trained on SCALED normal-only samples — matches notebook)
# ============================================================

normal_mask  = (y_train.values == 0)
X_normal_s   = X_train_s[normal_mask]

Xn_train_s, Xn_val_s = train_test_split(
    X_normal_s, test_size=0.20, random_state=42
)

print(f"\nAE train (normal): {Xn_train_s.shape}")
print(f"AE val   (normal): {Xn_val_s.shape}")

input_dim    = Xn_train_s.shape[1]           # 49
encoding_dim = max(8, input_dim // 2)        # 24

ae = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(encoding_dim,             activation="relu"),
    layers.Dense(max(4, encoding_dim // 2), activation="relu"),
    layers.Dense(encoding_dim,             activation="relu"),
    layers.Dense(input_dim,               activation="linear"),
])

ae.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse")
ae.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

print("\nTraining Autoencoder …")
ae.fit(
    Xn_train_s, Xn_train_s,
    validation_data=(Xn_val_s, Xn_val_s),
    epochs=15,
    batch_size=512,
    callbacks=[early_stop],
    verbose=1,
)


# ============================================================
# 8. AUTOENCODER THRESHOLD  (99th percentile of normal val MSE)
# ============================================================

recon_val = ae.predict(Xn_val_s, batch_size=2048, verbose=1)
mse_val   = np.mean(np.square(recon_val - Xn_val_s), axis=1)

ae_threshold = float(np.percentile(mse_val, 99))

print(f"\nAutoencoder threshold (99th pct normal val): {ae_threshold:.8f}")
print(f"Normal val MSE mean  : {mse_val.mean():.6f}")
print(f"Normal val MSE median: {np.median(mse_val):.6f}")


# ============================================================
# 9. AE TEST EVALUATION
# ============================================================

recon_test = ae.predict(X_test_s, batch_size=2048, verbose=1)
mse_test   = np.mean(np.square(recon_test - X_test_s), axis=1)
ae_preds   = (mse_test > ae_threshold).astype(np.int8)

print("\nAutoencoder Results")
print("--------------------------------")
print(f"Threshold: {ae_threshold:.8f}")
print(f"Accuracy : {accuracy_score(y_test, ae_preds):.6f}")
print(f"Precision: {precision_score(y_test, ae_preds, pos_label=1, zero_division=0):.6f}")
print(f"Recall   : {recall_score(y_test, ae_preds, pos_label=1, zero_division=0):.6f}")
print(f"F1 Score : {f1_score(y_test, ae_preds, pos_label=1, zero_division=0):.6f}")


# ============================================================
# 10. SAVE ARTIFACTS
# ============================================================

print("\nSaving artifacts …")

# Random Forest (unscaled)
joblib.dump(rf, RF_PATH)
print(f"  Saved RF      -> {RF_PATH}")

# Autoencoder
ae.save(AE_PATH)
print(f"  Saved AE      -> {AE_PATH}")

# Scaler
joblib.dump(scaler, SCALER_PATH)
print(f"  Saved Scaler  -> {SCALER_PATH}")

# ae_mse_max_train: used in live inference for combined score normalization
# Use the 99.9th percentile of test MSE so extreme outliers don't compress
# the combined score to near-zero for normal traffic.
ae_mse_max_train = float(np.percentile(mse_test, 99.9))

artifacts = {
    "feature_cols"      : FEATURE_COLS,
    "ae_threshold"      : ae_threshold,
    "ae_mse_max_train"  : ae_mse_max_train,
}

with open(ARTIFACTS_PATH, "w") as f:
    json.dump(artifacts, f, indent=2)

print(f"  Saved Artifacts -> {ARTIFACTS_PATH}")
print(f"    ae_threshold     = {ae_threshold:.8f}")
print(f"    ae_mse_max_train = {ae_mse_max_train:.8f}")

print("\nDONE: Training complete. Run  python app.py  to start the live dashboard.")
