import pytest

from app.services.document_processor import chunk_text, split_sentences

S1 = (
    "3.3 A variance of more than 10 percent requires a written justification "
    "and approval by the Chief Financial Officer (CFO) within 10 business days."
)
S2 = "3.4 Any single transaction above USD 10,000 requires approval by two authorized approvers."
S3 = (
    "4. Cloud and Software Spending 4.1 Cloud services and software license "
    "spending is reviewed monthly by the IT Director and the Finance Controller."
)
S4 = (
    "5.4 A transaction using a department and category combination that has "
    "not been used before must be confirmed by the department head."
)
POLICY = " ".join([S1, S2, S3, S4])


def test_heading_number_stays_with_its_title():
    sentences = split_sentences(POLICY)

    assert any(s.startswith("4. Cloud and Software Spending") for s in sentences)
    assert "4." not in sentences


def test_chunks_end_at_sentence_boundaries_and_respect_the_limit():
    chunks = chunk_text(POLICY, chunk_size=300, overlap=0)

    assert len(chunks) > 1
    assert all(chunk.endswith(".") for chunk in chunks)
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert all(chunk in POLICY for chunk in chunks)  # whole text, nothing altered


def test_overlap_repeats_the_last_whole_sentence():
    chunks = chunk_text(POLICY, chunk_size=300, overlap=150)

    assert chunks[1].startswith(S2)


def test_very_long_sentence_is_split_by_words():
    long_sentence = " ".join(["alpha"] * 100) + "."
    chunks = chunk_text(long_sentence, chunk_size=200, overlap=0)

    assert len(chunks) >= 3
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert len(" ".join(chunks).split()) == 100


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("Hello world.", chunk_size=100, overlap=100)