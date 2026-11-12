import re
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# A sentence ends at ".", "!" or "?" followed by whitespace.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A bare heading or clause number such as "4." that must stay attached
# to the title that follows it.
HEADING_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?$")


def extract_pdf_pages(file_content: bytes) -> list[dict]:
    """Extract text from a PDF while preserving page numbers."""

    if not file_content:
        raise ValueError("Uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(file_content))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            # Normalize whitespace without removing paragraph boundaries.
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                })

        if not pages:
            raise ValueError(
                "No extractable text found in this PDF. "
                "It may be a scanned document."
            )

        return pages

    except PdfReadError as exc:
        raise ValueError("Invalid or corrupted PDF file.") from exc


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping heading numbers with their titles."""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences: list[str] = []

    for part in SENTENCE_SPLIT.split(text):
        if sentences and HEADING_NUMBER.match(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)

    return sentences


def _split_long_sentence(sentence: str, limit: int) -> list[str]:
    """Fallback for a single sentence longer than a whole chunk."""

    pieces: list[str] = []
    current = ""

    for word in sentence.split(" "):
        candidate = f"{current} {word}".strip()

        if len(candidate) > limit and current:
            pieces.append(current)
            current = word
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


def _overlap_tail(sentences: list[str], overlap: int) -> list[str]:
    """Trailing whole sentences totalling at most `overlap` characters."""

    tail: list[str] = []
    size = 0

    for sentence in reversed(sentences):
        extra = len(sentence) + (1 if tail else 0)

        if size + extra > overlap:
            break

        tail.insert(0, sentence)
        size += extra

    return tail


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Pack whole sentences into chunks of at most `chunk_size` characters.

    A chunk always ends at a sentence boundary. The overlap is made of
    whole trailing sentences, so no sentence is ever cut in half.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    pieces: list[str] = []

    for sentence in split_sentences(text):
        if len(sentence) <= chunk_size:
            pieces.append(sentence)
        else:
            pieces.extend(_split_long_sentence(sentence, chunk_size))

    chunks: list[str] = []
    current: list[str] = []

    for piece in pieces:
        if current and len(" ".join(current + [piece])) > chunk_size:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap)

            # The overlap must never push the next chunk over the limit.
            while current and len(" ".join(current + [piece])) > chunk_size:
                current.pop(0)

        current.append(piece)

    if current:
        chunks.append(" ".join(current))

    return chunks


def process_pdf(file_content: bytes) -> list[dict]:
    """Extract a PDF and create page-aware text chunks."""

    pages = extract_pdf_pages(file_content)

    all_chunks = []

    for page in pages:
        page_chunks = chunk_text(page["text"])

        for chunk_index, chunk in enumerate(page_chunks):
            all_chunks.append({
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": chunk,
            })

    if not all_chunks:
        raise ValueError("PDF did not produce any text chunks.")

    return all_chunks