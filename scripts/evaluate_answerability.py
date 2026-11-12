"""Check that the answerability gate refuses unanswerable questions
and never refuses the labelled answerable ones.

Run from the project root:
    python scripts\\evaluate_answerability.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_retrieval import LABELLED, OFF_TOPIC, UNANSWERABLE  # noqa: E402
from app.services.answerability import check_coverage, is_answerable  # noqa: E402
from app.services.copilot_graph import MIN_RELEVANCE  # noqa: E402
from app.services.retrieval_service import search_documents  # noqa: E402


def judge(question: str) -> tuple[bool, float, list[str]]:
    results = search_documents(question=question, top_k=3)["results"]
    passages = [r["text"] for r in results if r["relevance_score"] >= MIN_RELEVANCE]

    if not passages:  # nothing retrieved at all: treated as not found
        return False, 0.0, ["(no passage above the relevance cutoff)"]

    coverage, missing = check_coverage(question, passages)
    return is_answerable(coverage, missing), coverage, missing


def run(title: str, questions: list[str], should_answer: bool) -> int:
    print(f"\n=== {title} ===")
    wrong = 0

    for question in questions:
        answerable, coverage, missing = judge(question)
        ok = answerable == should_answer
        wrong += 0 if ok else 1
        label = "answers " if answerable else "NOT FOUND"
        flag = "  " if ok else "!!"
        print(f"{flag} {label} coverage {coverage:.0%} missing {missing} | {question}")

    return wrong


def main() -> None:
    false_refusals = run("Labelled questions (must answer)", [q for q, _ in LABELLED], True)
    run("Unanswerable questions (must say not found)", UNANSWERABLE, False)
    run("Off-topic questions (must say not found)", OFF_TOPIC, False)
    print(f"\nFalse refusals among labelled questions: {false_refusals}/{len(LABELLED)}")


if __name__ == "__main__":
    main()