from app.services.answerability import check_coverage, is_answerable
from app.services.copilot_graph import synthesize_node

POLICY = (
    "3.3 A variance of more than 10 percent requires a written justification "
    "and approval by the Chief Financial Officer (CFO) within 10 business days. "
    "6.1 Items that remain unresolved after 10 business days are escalated to "
    "the CFO. 5.2 An anomaly flag is a trigger for review. It is not evidence "
    "of wrongdoing."
)


def test_topic_missing_from_the_text_is_not_answerable():
    coverage, missing = check_coverage("what is the CFO salary", [POLICY])

    assert missing == ["salary"]
    assert coverage == 0.5
    assert not is_answerable(coverage, missing)


def test_year_missing_from_the_text_is_not_answerable():
    text = "Service revenue $2,750 for the year ended September 30, 2021"
    coverage, missing = check_coverage(
        "What was the company's revenue in 2025?", [text]
    )

    assert "2025" in missing
    assert not is_answerable(coverage, missing)


def test_answerable_question_passes():
    coverage, missing = check_coverage(
        "Who must approve a variance above 10 percent?", [POLICY]
    )

    assert missing == []
    assert is_answerable(coverage, missing)


def test_different_word_forms_still_match():
    coverage, missing = check_coverage(
        "Are payments flagged?", ["A flag on a payment is reviewed."]
    )

    assert missing == []


def test_question_with_only_generic_words_is_answerable():
    assert check_coverage("What does the policy say?", [POLICY]) == (1.0, [])


def test_unanswerable_document_question_says_so_and_skips_the_llm(monkeypatch):
    def llm_must_not_run(*args, **kwargs):
        raise AssertionError("The LLM must not be called.")

    monkeypatch.setattr(
        "app.services.copilot_graph.chat_completion", llm_must_not_run
    )

    result = synthesize_node(
        {
            "question": "what is the CFO salary",
            "routes": ["rag_node"],
            "doc_text": "[SOURCE 1]\nContent:\n" + POLICY,
            "source_texts": [POLICY],
            "excerpts": ["[SOURCE 1] policy.pdf, page 1: " + POLICY],
            "sources": [],
            "warnings": [],
        }
    )

    assert "NOT FOUND IN THE DOCUMENTS" in result["answer"]
    assert "'salary'" in result["answer"]
    assert "CLOSEST PASSAGES" in result["answer"]