import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = set(
    """a about above after again all also am an and any are as at be because
    been before being below between both but by can could did do does doing
    down during each few for from further had has have having he her here
    hers him his how i if in into is it its itself just me more most my no
    nor not now of off on once only or other our out over own same she
    should so some such than that the their them then there these they this
    those through to too under until up very was we were what when where
    which while who whom why will with would you your
    company policy policies document documents uploaded information info
    data say says tell give show much many long soon often happen happens
    need needs needed row spent please list summarize summary describe
    mention mentioned state states according must may might shall get got
    like want know one new latest""".split()
)


def distinctive_terms(question: str) -> list[str]:
    """The question's meaningful words, in order, without duplicates."""
    terms: list[str] = []

    for token in TOKEN_PATTERN.findall(question.lower()):
        if len(token) < 2 or token in STOPWORDS or token in terms:
            continue
        terms.append(token)

    return terms


def _key(word: str) -> str:
    """First four letters, so approve/approval/approved all match."""
    return word[:4]


def check_coverage(
    question: str,
    passages: list[str],
) -> tuple[float, list[str]]:
    """Share of the question's key terms found in the passages.

    Returns (coverage from 0 to 1, the terms that were not found).
    """
    terms = distinctive_terms(question)

    if not terms:
        return 1.0, []

    vocabulary = {
        _key(word)
        for word in TOKEN_PATTERN.findall(" ".join(passages).lower())
    }

    missing = [term for term in terms if _key(term) not in vocabulary]

    return 1 - len(missing) / len(terms), missing


def is_answerable(coverage: float, missing: list[str]) -> bool:
    """Not answerable when half or more of the key terms are absent."""
    return not (missing and coverage <= 0.5)