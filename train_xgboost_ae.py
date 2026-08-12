"""
train_xgboost_ae.py
===================
AI-Powered Network Intrusion Detection System — XGBoost Training Pipeline
Phase 1: XGBoost + Autoencoder

Trains an XGBoost binary classifier on the NF-UNSW-NB15-v3 dataset using the
canonical 49-feature schema from feature_schema.py.

Artifacts saved:
    models/xgb_model.pkl        — trained XGBoostClassifier (unscaled input)
    models/xgb_artifacts.json   — feature_cols, xgb_ae_mse_max

Artifacts NOT touched:
    models/rf_model.pkl
    models/ae_model.keras
    models/scaler.pkl
    models/artifacts.json

Reused from existing training (must already exist):
    models/ae_model.keras   — existing Autoencoder (compatible: same 49 features)
    models/scaler.pkl       — existing StandardScaler (fitted on training data)
    models/artifacts.json   — existing AE threshold

Data-leakage policy:
    - XGBoost is fitted on X_train only.
    - xgb_ae_mse_max is derived from validation reconstruction errors (not test).
    - The test set is used solely in the final evaluation block.

Run once after train_models.py has completed:
    python train_xgboost_ae.py

Start the live dashboard (RF + XGBoost + AE all active):
    python app.py
"""

import os
import json
import random
import time
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
import xgboost as xgb

from feature_schema import FEATURE_COLS

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_HERE, "NF-UNSW-NB15-v3.csv")
MODELS_DIR   = os.path.join(_HERE, "models")

AE_PATH         = os.path.join(MODELS_DIR, "ae_model.keras")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
RF_ARTIFACTS    = os.path.join(MODELS_DIR, "artifacts.json")

XGB_PATH        = os.path.join(MODELS_DIR, "xgb_model.pkl")
XGB_ART_PATH    = os.path.join(MODELS_DIR, "xgb_artifacts.json")

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 0. PRE-FLIGHT CHECKS
# ============================================================

for path, label in [
    (DATASET_PATH, "NF-UNSW-NB15-v3.csv"),
    (AE_PATH,      "models/ae_model.keras"),
    (SCALER_PATH,  "models/scaler.pkl"),
    (RF_ARTIFACTS, "models/artifacts.json"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            f"Ensure '{label}' exists before running this script.\n"
            "Run  python train_models.py  first."
        )

print(f"\nFeature count : {len(FEATURE_COLS)}  (from feature_schema.py)")
assert len(FEATURE_COLS) == 49

# ── Load existing artifacts for AE threshold ─────────────────────────────────
with open(RF_ARTIFACTS) as f:
    rf_art = json.load(f)

AE_THRESHOLD = float(rf_art["ae_threshold"])
print(f"Existing AE threshold (reused): {AE_THRESHOLD:.8f}")


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset …")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape}")


# ============================================================
# 2. BINARY TARGET  (Benign = 0, Attack = 1)
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
# 4. TRAIN / TEST SPLIT  (same seed as train_models.py)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")


# ============================================================
# 5. LOAD EXISTING SCALER  (fitted on training data — no re-fitting)
# ============================================================

scaler = joblib.load(SCALER_PATH)
print("\nScaler loaded (not re-fitted — preserving existing artifact).")

X_train_s = scaler.transform(X_train).astype(np.float32)
X_test_s  = scaler.transform(X_test).astype(np.float32)


# ============================================================
# 6. LOAD EXISTING AUTOENCODER  (no re-training)
# ============================================================

print("\nLoading existing Autoencoder (not re-trained) …")
ae = tf.keras.models.load_model(AE_PATH)
print("Autoencoder loaded.")


# ============================================================
# 7. DERIVE xgb_ae_mse_max  (validation set ONLY — no leakage)
#
#    The existing artifacts.json stores ae_mse_max_train which was
#    computed from the test set (acknowledged leakage in the baseline).
#    This pipeline corrects that: we derive the normalization ceiling
#    from a held-out validation split of the TRAINING data only.
# ============================================================

print("\nDeriving AE MSE ceiling from training validation split (no leakage) …")

# Use a 20% validation split of X_train_s (normal + attack — full distribution)
X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
    X_train_s, y_train.values,
    test_size=0.20,
    random_state=42,
    stratify=y_train.values,
)

recon_val2   = ae.predict(X_val2, batch_size=2048, verbose=0)
mse_val2     = np.mean(np.square(recon_val2 - X_val2), axis=1)

# 99.9th percentile of validation MSE — captures extreme anomalies without
# compressing normal traffic scores toward zero.  Derived from training data
# only, so the test set remains completely unseen.
xgb_ae_mse_max = float(np.percentile(mse_val2, 99.9))

print(f"  xgb_ae_mse_max (99.9th pct, val): {xgb_ae_mse_max:.8f}")
print(
    "\n[Leakage note] The existing artifacts.json uses ae_mse_max_train derived "
    "from test MSE. This pipeline derives xgb_ae_mse_max from the validation "
    "set only — the test set is kept unseen until the evaluation block below."
)


# ============================================================
# 8. XGBOOST TRAINING
# ============================================================

# Class imbalance: use scale_pos_weight to balance attack vs normal.
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
spw   = n_neg / max(n_pos, 1)

print(f"\nTraining XGBoost …")
print(f"  neg={n_neg:,}  pos={n_pos:,}  scale_pos_weight={spw:.3f}")

xgb_model = xgb.XGBClassifier(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = spw,
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)

t0 = time.perf_counter()
xgb_model.fit(X_train, y_train)    # unscaled — tree models are scale-invariant
xgb_train_s = time.perf_counter() - t0
print(f"  XGBoost trained in {xgb_train_s:.1f}s")


# ============================================================
# 9. EXISTING RF RESULTS FOR COMPARISON
#    (load rf_model.pkl and evaluate on the SAME test set)
# ============================================================

print("\n" + "=" * 60)
print("EXISTING RANDOM FOREST RESULTS  (baseline comparison)")
print("=" * 60)

rf_model  = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
rf_att_idx = int(np.where(rf_model.classes_ == 1)[0][0])

t0        = time.perf_counter()
rf_preds  = rf_model.predict(X_test)
rf_probs  = rf_model.predict_proba(X_test)[:, rf_att_idx]
rf_lat_s  = (time.perf_counter() - t0) / len(X_test)

rf_acc  = accuracy_score(y_test, rf_preds)
rf_pre  = precision_score(y_test, rf_preds, pos_label=1, zero_division=0)
rf_rec  = recall_score(y_test, rf_preds, pos_label=1, zero_division=0)
rf_f1   = f1_score(y_test, rf_preds, pos_label=1, zero_division=0)
rf_auc  = roc_auc_score(y_test, rf_probs)
rf_prauc = average_precision_score(y_test, rf_probs)
rf_cm   = confusion_matrix(y_test, rf_preds)
rf_tn, rf_fp, rf_fn, rf_tp = rf_cm.ravel()
rf_fpr  = rf_fp / max(rf_fp + rf_tn, 1)

print(f"  Accuracy         : {rf_acc:.6f}")
print(f"  Precision        : {rf_pre:.6f}")
print(f"  Recall           : {rf_rec:.6f}")
print(f"  F1               : {rf_f1:.6f}")
print(f"  ROC-AUC          : {rf_auc:.6f}")
print(f"  PR-AUC           : {rf_prauc:.6f}")
print(f"  False Positive Rate: {rf_fpr:.6f}")
print(f"  Inference latency: {rf_lat_s*1e6:.2f} µs/flow")
print(f"  Confusion Matrix :\n    TN={rf_tn}  FP={rf_fp}\n    FN={rf_fn}  TP={rf_tp}")
print("\nClassification Report (RF):")
print(classification_report(y_test, rf_preds,
                             target_names=["Normal", "Attack"],
                             digits=4, zero_division=0))


# ============================================================
# 10. XGBOOST EVALUATION
# ============================================================

print("=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

xgb_att_idx = int(list(xgb_model.classes_).index(1))

t0        = time.perf_counter()
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, xgb_att_idx]
xgb_lat_s = (time.perf_counter() - t0) / len(X_test)

xgb_acc  = accuracy_score(y_test, xgb_preds)
xgb_pre  = precision_score(y_test, xgb_preds, pos_label=1, zero_division=0)
xgb_rec  = recall_score(y_test, xgb_preds, pos_label=1, zero_division=0)
xgb_f1   = f1_score(y_test, xgb_preds, pos_label=1, zero_division=0)
xgb_auc  = roc_auc_score(y_test, xgb_probs)
xgb_prauc = average_precision_score(y_test, xgb_probs)
xgb_cm   = confusion_matrix(y_test, xgb_preds)
xgb_tn, xgb_fp, xgb_fn, xgb_tp = xgb_cm.ravel()
xgb_fpr  = xgb_fp / max(xgb_fp + xgb_tn, 1)

print(f"  Accuracy         : {xgb_acc:.6f}")
print(f"  Precision        : {xgb_pre:.6f}")
print(f"  Recall           : {xgb_rec:.6f}")
print(f"  F1               : {xgb_f1:.6f}")
print(f"  ROC-AUC          : {xgb_auc:.6f}")
print(f"  PR-AUC           : {xgb_prauc:.6f}")
print(f"  False Positive Rate: {xgb_fpr:.6f}")
print(f"  Inference latency: {xgb_lat_s*1e6:.2f} µs/flow")
print(f"  Confusion Matrix :\n    TN={xgb_tn}  FP={xgb_fp}\n    FN={xgb_fn}  TP={xgb_tp}")
print("\nClassification Report (XGBoost):")
print(classification_report(y_test, xgb_preds,
                             target_names=["Normal", "Attack"],
                             digits=4, zero_division=0))


# ============================================================
# 11. AUTOENCODER EVALUATION (on test set — for reference)
# ============================================================

print("=" * 60)
print("AUTOENCODER RESULTS  (reused model, for reference)")
print("=" * 60)

t0         = time.perf_counter()
recon_test = ae.predict(X_test_s, batch_size=2048, verbose=0)
ae_lat_s   = (time.perf_counter() - t0) / len(X_test_s)

mse_test   = np.mean(np.square(recon_test - X_test_s), axis=1)
ae_preds   = (mse_test > AE_THRESHOLD).astype(np.int8)

ae_acc  = accuracy_score(y_test, ae_preds)
ae_pre  = precision_score(y_test, ae_preds, pos_label=1, zero_division=0)
ae_rec  = recall_score(y_test, ae_preds, pos_label=1, zero_division=0)
ae_f1   = f1_score(y_test, ae_preds, pos_label=1, zero_division=0)
ae_cm   = confusion_matrix(y_test, ae_preds)
ae_tn, ae_fp, ae_fn, ae_tp = ae_cm.ravel()
ae_fpr  = ae_fp / max(ae_fp + ae_tn, 1)

print(f"  Threshold        : {AE_THRESHOLD:.8f}")
print(f"  Accuracy         : {ae_acc:.6f}")
print(f"  Precision        : {ae_pre:.6f}")
print(f"  Recall           : {ae_rec:.6f}")
print(f"  F1               : {ae_f1:.6f}")
print(f"  False Positive Rate: {ae_fpr:.6f}")
print(f"  Inference latency: {ae_lat_s*1e6:.2f} µs/flow")
print(f"  Confusion Matrix :\n    TN={ae_tn}  FP={ae_fp}\n    FN={ae_fn}  TP={ae_tp}")


# ============================================================
# 12. XGBOOST + AE DIAGNOSTIC COMBINED SCORE
#
#     This is a simple Phase 1 diagnostic output only.
#     It is NOT the final fusion methodology — the proper
#     Confidence Engine / Dynamic Fusion will be implemented
#     in a later phase.
# ============================================================

print("=" * 60)
print("XGBoost + AE DIAGNOSTIC SCORE  (Phase 1 baseline only)")
print("=" * 60)
print(
    "  NOTE: xgb_score_combined = 0.5*xgb_prob + 0.5*ae_normalized\n"
    "  This is a simple diagnostic score for Phase 1 comparison.\n"
    "  The proper fusion mechanism will be implemented later.\n"
)

ae_norm_test       = np.clip(mse_test / (xgb_ae_mse_max + 1e-9), 0.0, 1.0)
xgb_score_combined = 0.5 * xgb_probs + 0.5 * ae_norm_test

# Threshold the diagnostic score at 0.5 for binary evaluation
xgb_ae_diag_preds  = (xgb_score_combined >= 0.5).astype(np.int8)
diag_acc  = accuracy_score(y_test, xgb_ae_diag_preds)
diag_pre  = precision_score(y_test, xgb_ae_diag_preds, pos_label=1, zero_division=0)
diag_rec  = recall_score(y_test, xgb_ae_diag_preds, pos_label=1, zero_division=0)
diag_f1   = f1_score(y_test, xgb_ae_diag_preds, pos_label=1, zero_division=0)
diag_auc  = roc_auc_score(y_test, xgb_score_combined)
diag_prauc = average_precision_score(y_test, xgb_score_combined)
diag_cm   = confusion_matrix(y_test, xgb_ae_diag_preds)
diag_tn, diag_fp, diag_fn, diag_tp = diag_cm.ravel()
diag_fpr  = diag_fp / max(diag_fp + diag_tn, 1)

print(f"  Accuracy         : {diag_acc:.6f}")
print(f"  Precision        : {diag_pre:.6f}")
print(f"  Recall           : {diag_rec:.6f}")
print(f"  F1               : {diag_f1:.6f}")
print(f"  ROC-AUC (score)  : {diag_auc:.6f}")
print(f"  PR-AUC  (score)  : {diag_prauc:.6f}")
print(f"  False Positive Rate: {diag_fpr:.6f}")
print(f"  Confusion Matrix :\n    TN={diag_tn}  FP={diag_fp}\n    FN={diag_fn}  TP={diag_tp}")


# ============================================================
# 13. COMPARISON SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
rows = [
    ("RF",              rf_acc,   rf_pre,   rf_rec,   rf_f1,   rf_auc,   rf_fpr,   rf_lat_s),
    ("XGBoost",         xgb_acc,  xgb_pre,  xgb_rec,  xgb_f1,  xgb_auc,  xgb_fpr,  xgb_lat_s),
    ("AE",              ae_acc,   ae_pre,   ae_rec,   ae_f1,   float("nan"), ae_fpr, ae_lat_s),
    ("XGB+AE (diag)",   diag_acc, diag_pre, diag_rec, diag_f1, diag_auc, diag_fpr, float("nan")),
]
header = f"{'Model':<18} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ROC-AUC':>9} {'FPR':>8} {'Lat(µs)':>10}"
print(header)
print("-" * len(header))
for name, acc, pre, rec, f1v, auc, fpr, lat in rows:
    auc_s = f"{auc:.6f}" if not (isinstance(auc, float) and np.isnan(auc)) else "   N/A  "
    lat_s = f"{lat*1e6:>10.2f}" if not (isinstance(lat, float) and np.isnan(lat)) else "       N/A"
    print(f"{name:<18} {acc:>8.6f} {pre:>8.6f} {rec:>8.6f} {f1v:>8.6f} {auc_s:>9} {fpr:>8.6f} {lat_s}")


# ============================================================
# 14. SAVE ARTIFACTS
# ============================================================

print("\nSaving artifacts …")

joblib.dump(xgb_model, XGB_PATH)
print(f"  Saved XGBoost   -> {XGB_PATH}")

xgb_artifacts = {
    "feature_cols"    : FEATURE_COLS,
    "xgb_ae_mse_max"  : xgb_ae_mse_max,
    # The AE threshold is read from artifacts.json at inference time;
    # we store it here too for completeness and traceability.
    "ae_threshold_ref": AE_THRESHOLD,
    "leakage_note"    : (
        "xgb_ae_mse_max is the 99.9th percentile of AE MSE on a training "
        "validation split — NOT derived from the test set."
    ),
}

with open(XGB_ART_PATH, "w") as f:
    json.dump(xgb_artifacts, f, indent=2)

print(f"  Saved Artifacts -> {XGB_ART_PATH}")
print(f"    xgb_ae_mse_max = {xgb_ae_mse_max:.8f}")

print(
    "\nDONE: XGBoost training complete.\n"
    "   Run  python app.py  to start the live dashboard with RF + XGBoost + AE.\n"
    "\n[NOT modified]\n"
    "   models/rf_model.pkl\n"
    "   models/ae_model.keras\n"
    "   models/scaler.pkl\n"
    "   models/artifacts.json\n"
)
