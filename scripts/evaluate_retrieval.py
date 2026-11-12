"""Measure retrieval quality against a small labelled question set.

Run from the project root (so .env is found):
    python scripts\\evaluate_retrieval.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.services.copilot_graph import MIN_RELEVANCE  # noqa: E402
from app.services.retrieval_service import search_documents  # noqa: E402

TOP_K = 5

# (question, fragment that must appear in a retrieved passage)
LABELLED = [
    # Budget variance policy
    ("How many business days does the department head have to explain a variance between 5 and 10 percent?",
     "written explanation to Finance within 5 business days"),
    ("Who must approve a variance above 10 percent?",
     "Chief Financial Officer (CFO)"),
    ("What approval is needed for a single transaction above USD 10,000?",
     "two authorized approvers"),
    ("What must the IT Director provide if cloud spending exceeds its budget?",
     "usage report and a cost-optimization plan"),
    ("How soon must flagged transactions be reviewed by the Finance Controller?",
     "within 3 business days"),
    ("Do weekend payments need approval?",
     "Payments dated on a weekend above USD 5,000"),
    ("How long must variance records be retained?",
     "retained for 7 years"),
    ("What happens when a category's spending rises three months in a row?",
     "budget reforecast"),
    ("Is an anomaly flag proof of wrongdoing?",
     "not evidence of wrongdoing"),
    # Sample financial statements
    ("What is the company's net income for the year?",
     "Net income $945"),
    ("What was the retained earnings balance on September 30, 2021?",
     "Balance, September 30, 2021 $1,265"),
    ("What are the total current liabilities?",
     "Total Current Liabilities 895"),
    ("What was the service revenue?",
     "Service revenue $2,750"),
    ("How much was spent on wages?",
     "Wages expenses 1,200"),
]

# On topic, but the documents cannot answer them. Informational only.
UNANSWERABLE = [
    "What was the company's revenue in 2025?",
    "What is the CFO's salary?",
]

# Not about finance at all. These should score below MIN_RELEVANCE.
OFF_TOPIC = [
    "What is the capital of France?",
    "Give me a recipe for chocolate cake.",
    "Who won the football world cup?",
]


def normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def top_results(question: str) -> list[dict]:
    return search_documents(question=question, top_k=TOP_K)["results"]


def main() -> None:
    ranks: list[int | None] = []
    answerable_top_scores: list[float] = []

    print(f"=== Labelled questions (top_k={TOP_K}) ===")

    for question, fragment in LABELLED:
        results = top_results(question)
        wanted = normalise(fragment)

        rank = next(
            (i for i, r in enumerate(results, start=1)
             if wanted in normalise(r["text"])),
            None,
        )
        ranks.append(rank)

        top_score = results[0]["relevance_score"] if results else 0.0
        answerable_top_scores.append(top_score)

        status = f"rank {rank}" if rank else "MISS  "
        print(f"[{status:>6}] top score {top_score:.3f} | {question}")

    total = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1)
    hit3 = sum(1 for r in ranks if r is not None and r <= 3)
    hit5 = sum(1 for r in ranks if r is not None)
    mrr = sum(1 / r for r in ranks if r) / total

    print(f"\nHit@1: {hit1}/{total} = {hit1 / total:.1%}")
    print(f"Hit@3: {hit3}/{total} = {hit3 / total:.1%}")
    print(f"Hit@5: {hit5}/{total} = {hit5 / total:.1%}")
    print(f"MRR:   {mrr:.3f}")

    print("\n=== Unanswerable questions (informational) ===")
    for question in UNANSWERABLE:
        results = top_results(question)
        score = results[0]["relevance_score"] if results else 0.0
        print(f"top score {score:.3f} | {question}")

    print(f"\n=== Off-topic questions (cutoff MIN_RELEVANCE = {MIN_RELEVANCE}) ===")
    off_topic_scores = []
    for question in OFF_TOPIC:
        results = top_results(question)
        score = results[0]["relevance_score"] if results else 0.0
        off_topic_scores.append(score)
        verdict = "blocked" if score < MIN_RELEVANCE else "PASSES the cutoff"
        print(f"top score {score:.3f} -> {verdict} | {question}")

    print("\n=== Does the cutoff separate real from junk questions? ===")
    print(f"Lowest top score of an answerable question: {min(answerable_top_scores):.3f}")
    print(f"Highest top score of an off-topic question: {max(off_topic_scores):.3f}")


if __name__ == "__main__":
    main()