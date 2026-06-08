"""Persist scoring run snapshots for executive trend analysis."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_COLUMNS = [
    "scored_at",
    "n_orders",
    "high_risk_count",
    "high_risk_pct",
    "avg_risk_score",
    "source",
]


class ScoringHistory:
    """
    Append-only CSV store for batch and real-time scoring run summaries.

    Each record captures aggregate risk metrics at scoring time so the executive
    dashboard can plot a real 30-day trend instead of simulated data.
    """

    def __init__(
        self,
        history_path: Union[str, Path],
        high_risk_threshold: float = 0.30,
    ) -> None:
        self.history_path = Path(history_path)
        self.high_risk_threshold = high_risk_threshold
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> pd.DataFrame:
        if not self.history_path.exists():
            return pd.DataFrame(columns=HISTORY_COLUMNS)

        df = pd.read_csv(self.history_path)
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[HISTORY_COLUMNS]

    def _save(self, df: pd.DataFrame) -> None:
        df.to_csv(self.history_path, index=False)

    def record_run(
        self,
        probabilities: List[float],
        source: str = "api",
        scored_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Persist aggregate metrics for a scoring run.

        Args:
            probabilities: Late probabilities returned by the model.
            source: Origin label (api, batch_pipeline, dashboard).
            scored_at: Optional override timestamp (UTC).

        Returns:
            Dict of the recorded snapshot.
        """
        if not probabilities:
            return {}

        timestamp = scored_at or datetime.now(timezone.utc)
        n_orders = len(probabilities)
        high_risk_count = sum(1 for p in probabilities if p >= self.high_risk_threshold)
        high_risk_pct = high_risk_count / n_orders
        avg_risk_score = sum(probabilities) / n_orders

        row = {
            "scored_at": timestamp.isoformat(),
            "n_orders": n_orders,
            "high_risk_count": high_risk_count,
            "high_risk_pct": round(high_risk_pct, 4),
            "avg_risk_score": round(avg_risk_score, 4),
            "source": source,
        }

        history = self._load()
        history.loc[len(history)] = row
        self._save(history)

        logger.info(
            "scoring_history_recorded scored_at=%s n_orders=%s high_risk_pct=%.4f source=%s",
            row["scored_at"],
            n_orders,
            high_risk_pct,
            source,
        )
        return row

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent scoring snapshot, if any."""
        history = self._load()
        if history.empty:
            return None
        latest = history.iloc[-1].to_dict()
        return {key: latest[key] for key in HISTORY_COLUMNS}

    def get_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Return daily high-risk percentage trend for the last N days.

        Multiple runs on the same day are averaged.
        """
        history = self._load()
        if history.empty:
            return []

        history["scored_at"] = pd.to_datetime(history["scored_at"], utc=True)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = history[history["scored_at"] >= cutoff].copy()
        if recent.empty:
            return []

        recent["date"] = recent["scored_at"].dt.date
        daily = (
            recent.groupby("date", as_index=False)["high_risk_pct"]
            .mean()
            .sort_values("date")
        )

        return [
            {"date": row["date"].isoformat(), "high_risk_pct": round(float(row["high_risk_pct"]), 4)}
            for _, row in daily.iterrows()
        ]

    def get_summary(self, trend_days: int = 30) -> Dict[str, Any]:
        """Build API response for dashboard and monitoring consumers."""
        trend = self.get_trend(days=trend_days)
        return {
            "trend_days": trend_days,
            "points": trend,
            "latest": self.get_latest(),
            "run_count": len(self._load()),
            "data_source": "persisted" if trend else "empty",
        }
