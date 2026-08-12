"""
inference.py
============
AI-Powered Network Traffic Analyzer — INFERENCE ENGINE

Loads saved artifacts from models/ and exposes run_inference().

run_inference(feature_dict) → result_dict

The feature_dict must contain keys matching FEATURE_COLS (the 49 model features)
plus the dashboard metadata keys prefixed with '_'.

Returns a dict with:
    rf_pred, rf_prob, ae_mse, ae_pred, score_combined     (existing RF+AE path)
    xgb_pred, xgb_prob, xgb_score_combined                (new XGB+AE path, if loaded)
    plus all dashboard metadata fields.

Feature schema is imported from feature_schema.py — the single canonical source.
"""

import os
import json
import logging
import datetime

import numpy as np
import pandas as pd

from feature_schema import FEATURE_COLS

logger = logging.getLogger(__name__)

_MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
_RF_PATH      = os.path.join(_MODELS_DIR, "rf_model.pkl")
_AE_PATH      = os.path.join(_MODELS_DIR, "ae_model.keras")
_SCALER_PATH  = os.path.join(_MODELS_DIR, "scaler.pkl")
_ART_PATH     = os.path.join(_MODELS_DIR, "artifacts.json")

_XGB_PATH     = os.path.join(_MODELS_DIR, "xgb_model.pkl")
_XGB_ART_PATH = os.path.join(_MODELS_DIR, "xgb_artifacts.json")


class InferenceEngine:
    """
    Singleton-like inference engine.  Load once at startup, reuse for every flow.

    RF + AE path: always available after load().
    XGBoost path: available after load_xgb() — gracefully absent if model not yet trained.
    """

    def __init__(self):
        # ── RF + AE (existing) ───────────────────────────────────────────────
        self._rf            = None
        self._ae            = None
        self._scaler        = None
        self._ae_threshold  = None
        self._ae_mse_max    = None
        self._attack_idx    = None
        self._loaded        = False

        # ── XGBoost (new, optional) ──────────────────────────────────────────
        self._xgb               = None
        self._xgb_attack_idx    = None
        self._xgb_ae_mse_max    = None   # validation-derived; leakage-free
        self._xgb_loaded        = False

    # ── RF + AE public API (unchanged) ────────────────────────────────────────

    def load(self):
        """Load RF + AE + Scaler artifacts from disk. Call once before run_inference."""
        import joblib
        import tensorflow as tf

        for path in (_RF_PATH, _AE_PATH, _SCALER_PATH, _ART_PATH):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Artifact not found: {path}\n"
                    "Run  python train_models.py  first."
                )

        logger.info("Loading RF …")
        self._rf = joblib.load(_RF_PATH)

        logger.info("Loading AE …")
        self._ae = tf.keras.models.load_model(_AE_PATH)

        logger.info("Loading Scaler …")
        self._scaler = joblib.load(_SCALER_PATH)

        with open(_ART_PATH) as f:
            art = json.load(f)

        # Validate that stored feature list matches the canonical schema
        stored_cols = art.get("feature_cols", [])
        if stored_cols != FEATURE_COLS:
            logger.warning(
                "artifacts.json feature_cols differs from FEATURE_COLS in "
                "feature_schema.py.  Using feature_schema.py as authoritative source."
            )

        self._ae_threshold = float(art["ae_threshold"])
        self._ae_mse_max   = float(art["ae_mse_max_train"])

        # Index of attack class (1) in RF
        self._attack_idx = int(np.where(self._rf.classes_ == 1)[0][0])

        self._loaded = True
        logger.info(
            "RF+AE inference engine ready. "
            f"AE threshold={self._ae_threshold:.6f}  "
            f"AE mse_max={self._ae_mse_max:.6f}"
        )

    # ── XGBoost public API (new, optional) ────────────────────────────────────

    def load_xgb(self):
        """
        Load XGBoost model and its artifacts.  Soft-fails if files are absent
        (training has not been run yet) so app.py starts without XGBoost.

        Must be called AFTER load() because it reuses self._ae and self._scaler.
        """
        import joblib

        if not self._loaded:
            logger.warning(
                "load_xgb() called before load() — skipping XGBoost load."
            )
            return

        if not os.path.exists(_XGB_PATH):
            logger.info(
                "XGBoost model not found (%s). "
                "Run  python train_xgboost_ae.py  to train. "
                "Continuing without XGBoost.",
                _XGB_PATH,
            )
            return

        if not os.path.exists(_XGB_ART_PATH):
            logger.warning(
                "xgb_artifacts.json not found (%s). "
                "Skipping XGBoost load.",
                _XGB_ART_PATH,
            )
            return

        try:
            logger.info("Loading XGBoost model …")
            self._xgb = joblib.load(_XGB_PATH)

            with open(_XGB_ART_PATH) as f:
                xgb_art = json.load(f)

            # Verify schema consistency
            stored_cols = xgb_art.get("feature_cols", [])
            if stored_cols != FEATURE_COLS:
                logger.warning(
                    "xgb_artifacts.json feature_cols differs from FEATURE_COLS. "
                    "Using feature_schema.py as authoritative source."
                )

            self._xgb_ae_mse_max = float(xgb_art["xgb_ae_mse_max"])

            # Index of attack class (1) in XGBoost
            classes = list(self._xgb.classes_)
            self._xgb_attack_idx = int(classes.index(1))

            self._xgb_loaded = True
            logger.info(
                "XGBoost inference ready. "
                f"xgb_ae_mse_max={self._xgb_ae_mse_max:.6f}"
            )

        except Exception as exc:
            logger.warning("Failed to load XGBoost: %s — continuing without it.", exc)
            self._xgb_loaded = False

    # ── Feature validation ────────────────────────────────────────────────────

    def validate_features(self, feature_dict: dict) -> bool:
        """
        Check that all 49 expected features are present in feature_dict.
        Returns True if valid, False otherwise.
        """
        if not self._loaded:
            return False
        missing = [c for c in FEATURE_COLS if c not in feature_dict]
        if missing:
            logger.warning(f"Missing features: {missing}")
            return False
        return True

    # ── Main inference ────────────────────────────────────────────────────────

    def run_inference(self, feature_dict: dict) -> dict:
        """
        Run RF + AE (and optionally XGBoost) on a single flow feature dict.

        Parameters
        ----------
        feature_dict : dict
            Must contain all 49 FEATURE_COLS keys (floats) plus '_*' metadata.

        Returns
        -------
        dict with keys:
            timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
            pkt_count, byte_count, duration, pkt_rate,
            true_label,
            rf_pred, rf_prob, ae_mse, ae_pred, score_combined   ← existing
            xgb_pred, xgb_prob, xgb_score_combined              ← new (if XGB loaded)
        """
        if not self._loaded:
            raise RuntimeError("InferenceEngine.load() has not been called.")

        if not self.validate_features(feature_dict):
            raise ValueError("Feature dict is missing required columns.")

        # ── Build 1-row feature array (unscaled) ────────────────────────────
        x_raw = np.array(
            [[feature_dict[c] for c in FEATURE_COLS]],
            dtype=np.float32,
        )

        # Replace any inf / nan
        x_raw = np.where(np.isfinite(x_raw), x_raw, 0.0)

        # ── RF prediction (unscaled input — matches training) ─────────────────
        rf_prob = float(self._rf.predict_proba(x_raw)[0, self._attack_idx])
        rf_pred = int(self._rf.predict(x_raw)[0])

        # ── AE prediction (scaled input — shared scaler) ──────────────────────
        x_scaled = self._scaler.transform(x_raw).astype(np.float32)
        recon    = self._ae.predict(x_scaled, verbose=0)
        ae_mse   = float(np.mean(np.square(recon - x_scaled)))
        ae_pred  = int(ae_mse > self._ae_threshold)

        # ── RF + AE combined score (existing formula — unchanged) ─────────────
        ae_normalized  = min(ae_mse / (self._ae_mse_max + 1e-9), 1.0)
        score_combined = 0.5 * rf_prob + 0.5 * ae_normalized

        # ── Assemble result dict ───────────────────────────────────────────────
        ts = datetime.datetime.fromtimestamp(feature_dict["_timestamp"])

        result = {
            "timestamp"      : ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "src_ip"         : feature_dict["_src_ip"],
            "dst_ip"         : feature_dict["_dst_ip"],
            "src_port"       : feature_dict["_src_port"],
            "dst_port"       : feature_dict["_dst_port"],
            "protocol"       : feature_dict["_protocol"],
            "pkt_count"      : feature_dict["_pkt_count"],
            "byte_count"     : feature_dict["_byte_count"],
            "duration"       : round(feature_dict["_duration"], 4),
            "pkt_rate"       : round(feature_dict["_pkt_rate"], 4),
            # No ground truth in live mode
            "true_label"     : "N/A — Live Traffic",
            "true_label_enc" : -1,
            # RF + AE outputs (existing, unchanged)
            "rf_pred"        : rf_pred,
            "rf_prob"        : round(rf_prob, 6),
            "ae_mse"         : round(ae_mse, 6),
            "ae_pred"        : ae_pred,
            "score_combined" : round(score_combined, 6),
        }

        # ── XGBoost + AE (additive, only if model is loaded) ─────────────────
        if self._xgb_loaded:
            # XGBoost trains on unscaled features — same convention as RF
            xgb_prob = float(
                self._xgb.predict_proba(x_raw)[0, self._xgb_attack_idx]
            )
            xgb_pred = int(self._xgb.predict(x_raw)[0])

            # Diagnostic combined score: XGB probability + AE anomaly score.
            # NOTE: This is a simple baseline diagnostic for Phase 1 only.
            # The proper fusion mechanism will be implemented in a later phase
            # (Confidence Engine / Dynamic Fusion).
            xgb_ae_normalized    = min(ae_mse / (self._xgb_ae_mse_max + 1e-9), 1.0)
            xgb_score_combined   = 0.5 * xgb_prob + 0.5 * xgb_ae_normalized

            result["xgb_pred"]           = xgb_pred
            result["xgb_prob"]           = round(xgb_prob, 6)
            result["xgb_score_combined"] = round(xgb_score_combined, 6)

        return result


# ── Module-level singleton ────────────────────────────────────────────────────
engine = InferenceEngine()
