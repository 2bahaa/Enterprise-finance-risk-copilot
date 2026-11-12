"""Grade Isolation Forest against the synthetic answer key.

Run from the project root (so .env is found):
    python scripts\\evaluate_anomalies.py --dataset-id 2
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.database import SessionLocal  # noqa: E402
from app.services.anomaly_service import detect_financial_anomalies  # noqa: E402

GROUND_TRUTH_FILE = PROJECT_ROOT / "data" / "synthetic_ground_truth.csv"
SETTINGS = [0.01, 0.02, 0.03, 0.05, 0.10, "auto"]


def load_ground_truth() -> dict[str, str]:
    """Map transaction_id -> anomaly_type for the injected anomalies."""
    with GROUND_TRUTH_FILE.open(newline="", encoding="utf-8") as f:
        return {
            row["transaction_id"]: row["anomaly_type"]
            for row in csv.DictReader(f)
        }


def run_detection(
    dataset_id: int,
    contamination: float | str,
    engineered: bool,
) -> dict:
    with SessionLocal() as db:
        return detect_financial_anomalies(
            db=db,
            dataset_id=dataset_id,
            contamination=contamination,
            engineered_features=engineered,
        )


def evaluate(flagged: set[str], truth: dict[str, str]) -> dict:
    caught = len(flagged & set(truth))
    precision = caught / len(flagged) if flagged else 0.0
    recall = caught / len(truth)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "flagged": len(flagged),
        "caught": caught,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def report(dataset_id: int, truth: dict[str, str], engineered: bool) -> None:
    label = "ENGINEERED features" if engineered else "BASELINE features"
    print(f"\n=== {label} ===")

    header = (f"{'contamination':>13} | {'flagged':>7} | {'caught':>6} | "
              f"{'precision':>9} | {'recall':>6} | {'F1':>5}")
    print(header)
    print("-" * len(header))

    results = []

    for contamination in SETTINGS:
        result = run_detection(dataset_id, contamination, engineered)

        flagged = {
            item["transaction_id"]
            for item in result["anomalies"]
            if item["is_anomaly"]
        }
        metrics = evaluate(flagged, truth)
        results.append(result)

        print(f"{str(contamination):>13} | {metrics['flagged']:>7} | "
              f"{metrics['caught']:>6} | {metrics['precision']:>9.1%} | "
              f"{metrics['recall']:>6.1%} | {metrics['f1']:>5.2f}")

    # Ranking quality does not depend on the contamination setting:
    # it only moves the cutoff. Results are sorted by anomaly score.
    ranked_ids = [item["transaction_id"] for item in results[0]["anomalies"]]
    k = len(truth)
    hits = sum(1 for tid in ranked_ids[:k] if tid in truth)
    print(f"\nPrecision@{k} (top {k} by score): {hits}/{k} = {hits / k:.1%}")

    rank_of = {tid: pos for pos, tid in enumerate(ranked_ids, start=1)}
    missing_rank = len(ranked_ids) + 1

    print("\nRank of each injected anomaly (1 = most suspicious):")
    for tid, kind in sorted(
        truth.items(), key=lambda kv: rank_of.get(kv[0], missing_rank)
    ):
        print(f"  {tid}  {kind:<17} rank {rank_of.get(tid, missing_rank)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, required=True)
    args = parser.parse_args()

    truth = load_ground_truth()
    print(f"Ground truth: {len(truth)} injected anomalies")

    report(args.dataset_id, truth, engineered=False)
    report(args.dataset_id, truth, engineered=True)


if __name__ == "__main__":
    main()