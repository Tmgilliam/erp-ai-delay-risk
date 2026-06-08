"""
Bootstrap scoring history for executive dashboard trend demos.

Run once after setup to populate monitoring/scoring_history.csv with 30 days
of trend data derived from the bundled sample scoring file.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model import DelayRiskModel  # noqa: E402
from monitoring.scoring_history import ScoringHistory  # noqa: E402

SAMPLE_PATH = ROOT / "data" / "open_orders_scoring_sample.csv"
HISTORY_PATH = ROOT / "monitoring" / "scoring_history.csv"


def seed_history(days: int = 30) -> None:
    """Score sample data and backfill daily history with slight operational variation."""
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample data not found: {SAMPLE_PATH}")

    model = DelayRiskModel()
    sample = pd.read_csv(SAMPLE_PATH)
    batch_result = model.predict_batch(sample.to_dict(orient="records"))
    base_probs = [row["late_probability"] for row in batch_result["results"]]
    base_high_risk_pct = sum(1 for p in base_probs if p >= 0.30) / len(base_probs)

    history = ScoringHistory(HISTORY_PATH)
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)

    for offset in range(days, 0, -1):
        scored_at = now - timedelta(days=offset)
        # Simulate day-to-day operational variation around the real sample baseline.
        noise = float(rng.normal(0, 0.03))
        scaled_probs = [min(max(p + noise, 0.01), 0.99) for p in base_probs]
        history.record_run(
            probabilities=scaled_probs,
            source="seed",
            scored_at=scored_at,
        )

    print(f"Seeded {days} days of scoring history to {HISTORY_PATH}")
    print(f"Baseline high-risk % from sample: {base_high_risk_pct:.1%}")


if __name__ == "__main__":
    seed_history()
