# Model Drift Monitoring — ERP Delay Risk

## Why Drift Monitoring Matters in Enterprise ML

A model trained on last quarter's ERP patterns will silently degrade when operations change: new suppliers, shifted lead times, inventory policy updates, or seasonal demand spikes. Without drift monitoring, teams discover the problem only after fill rates drop or expedite costs spike — the same failure mode Dr. Gilliam saw in manufacturing environments where process drift has real P&L impact.

Deploying inference without drift detection is an enterprise AI governance anti-pattern. It signals a demo mindset, not production operations thinking.

## How This Implementation Works

`drift_monitor.py` implements a `DriftMonitor` class that:

1. Loads a **reference baseline** (`monitoring/reference_baseline.csv`) derived from training data
2. Compares **current scoring samples** against baseline per numeric feature
3. Runs a **Kolmogorov-Smirnov (KS) two-sample test** via `scipy.stats`
4. Flags features where **p-value < threshold** (default 0.05)
5. Logs every check with timestamp, feature name, p-value, and threshold context
6. Returns JSON + human-readable summary via `GET /monitoring/drift-report`

**Why scipy, not Evidently?** `evidently` is not in `requirements.txt`. scipy keeps the dependency footprint light while delivering statistically defensible drift detection for tabular ERP features. Evidently can be added later for richer report visualizations without changing the API contract.

## Production Integration — Azure Monitor

In the Azure target architecture, drift events flow like this:

```
Drift report JSON → Azure Function → custom metric → Azure Monitor alert → ops team notification
```

Concrete wiring:

- **FastAPI** exposes `/monitoring/drift-report` on a schedule (Logic App or Container Apps job)
- **Azure Function** parses `drift_flags`, emits `CustomMetrics` per drifted feature
- **Azure Monitor alert rule** fires when `drift_detected_count > 0`
- **Action Group** notifies operations + ML owner via Teams/email
- **Log Analytics** retains full drift JSON for audit and model governance review

## Operational Context

In healthcare manufacturing, Dr. Gilliam managed environments where 98% inventory accuracy and 95% fill rate were measured outcomes — not aspirational KPIs. Process drift in purchasing, lead time variance, or inventory positioning directly threatened those numbers. This monitoring module reflects floor-level understanding: when the data distribution shifts, the model's view of "normal" is wrong, and someone in operations needs to know before Monday's planning call.

## Files

| File | Purpose |
|------|---------|
| `drift_monitor.py` | KS-test drift detection module |
| `reference_baseline.csv` | Training-time feature distribution baseline |
| `README.md` | This document |

## Scoring History (Phase 3)

`scoring_history.py` persists aggregate metrics from every scoring run so the executive dashboard plots a **real 30-day trend** instead of simulated data.

| Endpoint | Purpose |
|----------|---------|
| `GET /monitoring/scoring-history` | Daily high-risk % trend for dashboard |
| Auto-record on `POST /predict` | Single-order runs logged |
| Auto-record on `POST /predict/batch` | Batch runs logged |

Bootstrap demo history:

```bash
python monitoring/seed_scoring_history.py
```

In Azure production, this CSV would migrate to Blob Storage or Log Analytics — same API contract, different persistence backend.

## Local Usage

```bash
# With API running on port 8001
curl http://127.0.0.1:8001/monitoring/drift-report
curl http://127.0.0.1:8001/monitoring/scoring-history
```

Or programmatically:

```python
from monitoring.drift_monitor import DriftMonitor
import pandas as pd

monitor = DriftMonitor("monitoring/reference_baseline.csv")
current = pd.read_csv("data/open_orders_scoring_sample.csv")
results = monitor.compute_drift(current)
flags = monitor.flag_drift(results, threshold=0.05)
report = monitor.generate_report(results, flags=flags)
print(report["summary"])
```
