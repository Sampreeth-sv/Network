"""
inference.py
============
AI-Powered Network Traffic Analyzer — INFERENCE ENGINE

Loads saved artifacts from models/ and exposes run_inference().

run_inference(feature_dict) → result_dict

The feature_dict must contain keys matching FEATURE_COLS (the 49 model features)
plus the dashboard metadata keys prefixed with '_'.

Returns a dict with:
    rf_pred, rf_prob, ae_mse, ae_pred, score_combined
    plus all dashboard metadata fields.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
_RF_PATH     = os.path.join(_MODELS_DIR, "rf_model.pkl")
_AE_PATH     = os.path.join(_MODELS_DIR, "ae_model.keras")
_SCALER_PATH = os.path.join(_MODELS_DIR, "scaler.pkl")
_ART_PATH    = os.path.join(_MODELS_DIR, "artifacts.json")


class InferenceEngine:
    """
    Singleton-like inference engine.  Load once at startup, reuse for every flow.
    """

    def __init__(self):
        self._rf            = None
        self._ae            = None
        self._scaler        = None
        self._feature_cols  = None
        self._ae_threshold  = None
        self._ae_mse_max    = None
        self._attack_idx    = None
        self._loaded        = False

    # ── public ──────────────────────────────────────────────────────────────

    def load(self):
        """Load all artifacts from disk. Call once before run_inference."""
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

        self._feature_cols = art["feature_cols"]
        self._ae_threshold = float(art["ae_threshold"])
        self._ae_mse_max   = float(art["ae_mse_max_train"])

        # Index of attack class (1) in RF
        self._attack_idx = int(np.where(self._rf.classes_ == 1)[0][0])

        self._loaded = True
        logger.info(
            "Inference engine ready. "
            f"AE threshold={self._ae_threshold:.6f}  "
            f"AE mse_max={self._ae_mse_max:.6f}"
        )

    def validate_features(self, feature_dict: dict) -> bool:
        """
        Check that all 49 expected features are present in feature_dict.
        Returns True if valid, False otherwise.
        """
        if not self._loaded:
            return False
        missing = [c for c in self._feature_cols if c not in feature_dict]
        if missing:
            logger.warning(f"Missing features: {missing}")
            return False
        return True

    def run_inference(self, feature_dict: dict) -> dict:
        """
        Run RF + AE on a single flow feature dict.

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
            rf_pred, rf_prob, ae_mse, ae_pred, score_combined
        """
        if not self._loaded:
            raise RuntimeError("InferenceEngine.load() has not been called.")

        if not self.validate_features(feature_dict):
            raise ValueError("Feature dict is missing required columns.")

        # ── Build 1-row feature array (unscaled) for RF ──────────────────────
        x_raw = np.array(
            [[feature_dict[c] for c in self._feature_cols]],
            dtype=np.float32,
        )

        # Replace any inf / nan
        x_raw = np.where(np.isfinite(x_raw), x_raw, 0.0)

        # ── RF prediction (unscaled input — matches training) ─────────────────
        rf_prob = float(self._rf.predict_proba(x_raw)[0, self._attack_idx])
        rf_pred = int(self._rf.predict(x_raw)[0])

        # ── AE prediction (scaled input — matches training) ───────────────────
        x_scaled = self._scaler.transform(x_raw).astype(np.float32)
        recon    = self._ae.predict(x_scaled, verbose=0)
        ae_mse   = float(np.mean(np.square(recon - x_scaled)))
        ae_pred  = int(ae_mse > self._ae_threshold)

        # ── Combined score (faithful to notebook formula) ──────────────────────
        # 0.5 * rf_prob + 0.5 * (ae_mse / ae_mse_max)
        ae_normalized    = ae_mse / (self._ae_mse_max + 1e-9)
        ae_normalized    = min(ae_normalized, 1.0)   # cap at 1
        score_combined   = 0.5 * rf_prob + 0.5 * ae_normalized

        # ── Assemble result dict ───────────────────────────────────────────────
        import datetime
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
            # Model outputs
            "rf_pred"        : rf_pred,
            "rf_prob"        : round(rf_prob, 6),
            "ae_mse"         : round(ae_mse, 6),
            "ae_pred"        : ae_pred,
            "score_combined" : round(score_combined, 6),
        }

        return result


# ── Module-level singleton ────────────────────────────────────────────────────
engine = InferenceEngine()
