"""
Batch scoring pipeline for ERP open-order delay risk.

Scores a CSV of open orders via the FastAPI inference service and writes
results to disk. Designed for scheduled runs (e.g., nightly open-order book).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "open_orders_scoring_sample.csv"
DEFAULT_OUTPUT = ROOT / "data" / "scored_orders_output.csv"
DEFAULT_API_URL = "http://127.0.0.1:8001"


def score_batch_via_api(
    records: List[Dict[str, Any]],
    api_url: str,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Call /predict/batch and return API JSON response."""
    endpoint = f"{api_url.rstrip('/')}/predict/batch"
    response = requests.post(endpoint, json=records, timeout=timeout)
    response.raise_for_status()
    return response.json()


def run_batch_score(
    input_path: Path,
    output_path: Path,
    api_url: str,
) -> Path:
    """
    Read orders CSV, score via API, merge predictions, and write output.

    Returns:
        Path to the scored output CSV.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    records = df.to_dict(orient="records")
    result = score_batch_via_api(records, api_url=api_url)

    predictions = pd.DataFrame(result["results"])
    scored = df.merge(predictions, on="order_id", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    print(
        f"Scored {result['n_orders']} orders — "
        f"{result['late_count']} flagged late — saved to {output_path}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score ERP orders via API.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV with ERP order payloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path for scored results.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="Base URL for FastAPI inference service.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_batch_score(args.input, args.output, args.api_url)
        return 0
    except requests.RequestException as exc:
        print(f"API request failed: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
