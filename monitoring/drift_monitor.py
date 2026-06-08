"""Feature drift detection for ERP delay risk model governance."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Compare incoming feature distributions against a training baseline.

    Uses the two-sample Kolmogorov-Smirnov test (scipy) for numeric features.
    Evidently is not in requirements.txt; scipy keeps the dependency footprint
    light while meeting enterprise drift detection needs for tabular ERP data.
    """

    def __init__(self, reference_data_path: Union[str, Path]) -> None:
        """
        Load training baseline distribution from reference CSV or Parquet.

        Args:
            reference_data_path: Path to baseline feature file used at training time.
        """
        path = Path(reference_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference data not found: {path}")

        if path.suffix.lower() == ".parquet":
            self.reference_data = pd.read_parquet(path)
        else:
            self.reference_data = pd.read_csv(path)

        # Drift checks apply to numeric operational drivers only.
        self.numeric_features: List[str] = self.reference_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()

    def compute_drift(self, current_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Compare current feature distributions against the reference baseline.

        Returns:
            Dict mapping feature name to KS statistic, p-value, and sample sizes.
        """
        results: Dict[str, Dict[str, Any]] = {}

        for feature in self.numeric_features:
            if feature not in current_data.columns:
                results[feature] = {
                    "ks_statistic": None,
                    "p_value": None,
                    "reference_n": len(self.reference_data),
                    "current_n": 0,
                    "status": "missing_in_current",
                }
                continue

            ref_series = self.reference_data[feature].dropna()
            cur_series = current_data[feature].dropna()

            if len(ref_series) < 2 or len(cur_series) < 2:
                results[feature] = {
                    "ks_statistic": None,
                    "p_value": None,
                    "reference_n": len(ref_series),
                    "current_n": len(cur_series),
                    "status": "insufficient_samples",
                }
                continue

            ks_stat, p_value = ks_2samp(ref_series, cur_series)
            results[feature] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "reference_n": int(len(ref_series)),
                "current_n": int(len(cur_series)),
                "status": "ok",
            }

            logger.info(
                "drift_check timestamp=%s feature=%s p_value=%.6f threshold_context=ks_test",
                datetime.now(timezone.utc).isoformat(),
                feature,
                p_value,
            )

        return results

    def flag_drift(
        self,
        drift_results: Dict[str, Dict[str, Any]],
        threshold: float = 0.05,
    ) -> Dict[str, bool]:
        """
        Flag features whose p-value falls below the significance threshold.

        A low p-value indicates the current distribution differs from baseline.
        """
        flags: Dict[str, bool] = {}

        for feature, metrics in drift_results.items():
            p_value = metrics.get("p_value")
            if p_value is None:
                flags[feature] = False
                continue

            drift_detected = p_value < threshold
            flags[feature] = drift_detected

            if drift_detected:
                logger.warning(
                    "drift_detected timestamp=%s feature=%s p_value=%.6f threshold=%.4f",
                    datetime.now(timezone.utc).isoformat(),
                    feature,
                    p_value,
                    threshold,
                )

        return flags

    def generate_report(
        self,
        drift_results: Dict[str, Dict[str, Any]],
        flags: Optional[Dict[str, bool]] = None,
        threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Build JSON report (for API) and human-readable summary string.

        Returns:
            Dict with timestamp, threshold, per-feature metrics, flags, and summary.
        """
        if flags is None:
            flags = self.flag_drift(drift_results, threshold=threshold)

        drifted_features = [f for f, detected in flags.items() if detected]
        checked_features = [
            f for f, m in drift_results.items() if m.get("status") == "ok"
        ]

        if not drifted_features:
            summary = (
                f"No statistically significant drift detected across "
                f"{len(checked_features)} numeric features (threshold={threshold})."
            )
        else:
            summary = (
                f"Drift detected on {len(drifted_features)} feature(s): "
                f"{', '.join(drifted_features)}. "
                f"Review operational drivers and consider model retraining."
            )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "kolmogorov_smirnov",
            "threshold": threshold,
            "features_checked": len(checked_features),
            "drift_detected_count": len(drifted_features),
            "drift_results": drift_results,
            "drift_flags": flags,
            "summary": summary,
            "summary_json": json.dumps(
                {
                    "drifted_features": drifted_features,
                    "threshold": threshold,
                }
            ),
        }
