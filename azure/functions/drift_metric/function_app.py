"""
Azure Function: emit drift alert custom metric to Azure Monitor.

Timer-triggered daily check of /monitoring/drift-report endpoint.
Deploy separately after Container Apps are live.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import azure.functions as func
import requests

app = func.FunctionApp()

API_BASE_URL = os.getenv("ERP_AI_API_URL", "").rstrip("/")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))


@app.timer_trigger(schedule="0 0 6 * * *", arg_name="timer", run_on_startup=False)
def drift_metric_timer(timer: func.TimerRequest) -> None:
    """Poll drift report and log metric payload for Azure Monitor ingestion."""
    if not API_BASE_URL:
        logging.error("ERP_AI_API_URL not configured")
        return

    response = requests.get(
        f"{API_BASE_URL}/monitoring/drift-report",
        params={"threshold": DRIFT_THRESHOLD},
        timeout=60,
    )
    response.raise_for_status()
    report = response.json()

    metric_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric_name": "ERP.DelayRisk.DriftDetectedCount",
        "metric_value": report.get("drift_detected_count", 0),
        "features_checked": report.get("features_checked", 0),
        "summary": report.get("summary", ""),
    }

    # In production, emit via Azure Monitor OpenTelemetry exporter or
    # Application Insights customEvents.track. Logged here for audit trail.
    logging.warning("DRIFT_METRIC %s", json.dumps(metric_payload))

    if metric_payload["metric_value"] > 0:
        logging.error(
            "Drift detected on %s feature(s) — review required",
            metric_payload["metric_value"],
        )
