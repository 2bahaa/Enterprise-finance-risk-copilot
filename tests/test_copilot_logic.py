import pytest

from app.services.copilot_graph import (
    choose_routes,
    detect_department,
    find_ungrounded_numbers,
    verbatim_support,
    verify_summary,
)

DEPARTMENTS = ["Finance", "HR", "IT", "Legal", "Marketing", "Operations"]

SOURCE = (
    "3.2 A variance of more than 5 percent and up to 10 percent requires the "
    "department head to submit a written explanation to Finance within 5 "
    "business days. 5.1 Transactions flagged by statistical anomaly detection "
    "must be reviewed by the Finance Controller within 3 business days."
)

EXACT_SENTENCE = (
    "Transactions flagged by statistical anomaly detection must be reviewed "
    "by the Finance Controller within 3 business days."
)


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What are the total assets?", ["rag_node"]),
        ("Which transactions look unusual?", ["anomaly_node"]),
        (
            "Which department spends the most and are there "
            "suspicious transactions?",
            ["kpi_node", "anomaly_node"],
        ),
        (
            "Why did IT spending increase, which transactions look "
            "unusual, and what does the budget policy say about this "
            "variance?",
            ["kpi_node", "variance_node", "anomaly_node", "rag_node"],
        ),
        ("Why did IT spending increase?", ["kpi_node", "variance_node"]),
        ("hello there", ["rag_node"]),  # nothing matched: default route
    ],
)
def test_choose_routes(question, expected):
    assert choose_routes(question) == expected


def test_detects_acronym_department():
    assert detect_department("Why did IT spending increase?", DEPARTMENTS) == "IT"


def test_pronoun_it_is_not_the_it_department():
    assert detect_department("Why did it increase?", DEPARTMENTS) is None


def test_detects_full_department_name_case_insensitively():
    assert detect_department("how is marketing doing", DEPARTMENTS) == "Marketing"


def test_unknown_department_returns_none():
    assert detect_department("How is Sales doing?", DEPARTMENTS) is None


def test_grounded_number_passes_even_with_different_formatting():
    evidence = "Marketing: USD 3,200.00 and total USD 8,401.50"
    assert find_ungrounded_numbers("Marketing spent $3,200.", evidence) == []


def test_invented_number_is_flagged():
    assert find_ungrounded_numbers("Total assets: $13,060", "Cash: $1,550") == ["13,060"]


def test_exact_copy_has_full_support():
    assert verbatim_support(EXACT_SENTENCE, SOURCE) == 1.0


def test_premise_echo_has_no_support():
    assert verbatim_support("The IT spending increased.", SOURCE) == 0.0


def test_completed_truncated_sentence_is_rejected():
    # The chunk was cut mid-sentence and the model invented the ending.
    source = "5.4 A transaction using a department and category combination that has not"
    statement = (
        "A transaction using a department and category combination that "
        "has not been approved requires two authorized approvers."
    )
    assert verbatim_support(statement, source) < 0.8


def test_short_statement_uses_word_coverage():
    assert verbatim_support("Cash: $1,550", "Cash $ 1,550 Accounts receivable 770") == 1.0


def test_verify_summary_keeps_verbatim_bullets_labels_them_and_drops_the_rest():
    summary = (
        "Here are the sentences:\n"
        "• The IT spending increased.\n"
        f"• {EXACT_SENTENCE}"
    )

    kept, removed = verify_summary(summary, ["unrelated text", SOURCE])

    assert removed == 1
    assert kept == [f"• {EXACT_SENTENCE} [SOURCE 2]"]