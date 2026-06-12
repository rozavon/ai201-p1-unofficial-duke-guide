"""Milestone 3 — Document ingestion and chunking for the Unofficial Duke Dorm Guide.

Implements the Chunking Strategy from planning.md:
  * 1 chunk = 1 review, split on the `---` delimiter between reviews.
  * dorm_name comes from the filename; review_id identifies the source review.
  * Token-limit guardrail: all-MiniLM-L6-v2 silently truncates past 256 tokens, so any
    review longer than that is sub-split on sentence boundaries with ~1 sentence of
    overlap. Sub-chunks of the same review share one review_id.
  * Zero overlap *between* different reviews — they are distinct opinions.

This module only covers ingestion + chunking (Milestone 3). Embedding/retrieval
(Milestone 4) and generation (Milestone 5) live elsewhere.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# all-MiniLM-L6-v2 max sequence length. The model silently truncates beyond this,
# so it is also the threshold at which we sub-split a single review.
MAX_TOKENS = 256

# Directory holding one .txt file per dorm, relative to the repo root.
DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"

# Matches a delimiter line that contains only dashes (e.g. "---"), optionally padded.
_DELIMITER_RE = re.compile(r"(?m)^\s*-{3,}\s*$")

# Splits on whitespace that follows sentence-ending punctuation. Good enough for the
# short, informal reviews in this corpus (used only inside the long-review sub-split path).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Tokenizer (prefer the real model tokenizer; fall back to a word heuristic)
# --------------------------------------------------------------------------- #

_tokenizer = None
_tokenizer_loaded = False


def _get_tokenizer():
    """Lazily load the all-MiniLM-L6-v2 tokenizer.

    Returns None if transformers/sentence-transformers isn't installed yet, in which
    case _count_tokens() falls back to the word-based heuristic from planning.md.
    """
    global _tokenizer, _tokenizer_loaded
    if _tokenizer_loaded:
        return _tokenizer
    _tokenizer_loaded = True
    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception:
        _tokenizer = None
    return _tokenizer


def _count_tokens(text: str) -> int:
    """Token count as the embedding model would see it (including special tokens).

    Falls back to planning.md's heuristic (~200 words ≈ 256 tokens, i.e. ~1.3 tokens
    per word) when the real tokenizer isn't available.
    """
    tok = _get_tokenizer()
    if tok is not None:
        return len(tok.encode(text))
    return round(len(text.split()) * 1.3)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def _clean_text(text: str) -> str:
    """Conservative cleaning that preserves review content and paragraph breaks.

    - Normalize line endings to "\\n".
    - Strip trailing whitespace from each line.
    - Collapse 3+ consecutive newlines down to a single blank line (one paragraph break).
    - Strip leading/trailing whitespace overall.

    We intentionally keep single blank lines: some reviews use real paragraph breaks,
    and those boundaries are useful when sub-splitting a long review.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_documents(documents_dir: os.PathLike | str = DOCUMENTS_DIR) -> list[dict]:
    """Read every .txt file in ``documents_dir`` with Python's built-in file reader.

    Returns a list of ``{"dorm_name": str, "text": str}`` dicts, sorted by filename
    for deterministic output. ``dorm_name`` is the filename stem (e.g. "basset.txt"
    -> "basset"). Text is cleaned via :func:`_clean_text`.
    """
    documents_dir = Path(documents_dir)
    docs: list[dict] = []
    for path in sorted(documents_dir.glob("*.txt")):
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        docs.append({"dorm_name": path.stem, "text": _clean_text(raw)})
    return docs


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def _split_long_review(text: str) -> list[str]:
    """Sub-split a review that exceeds MAX_TOKENS on sentence boundaries.

    Greedily packs sentences up to the token budget. Each new sub-chunk begins with
    the last sentence of the previous one (~1 sentence of overlap) so a sentence that
    straddles the boundary isn't lost from either embedding.
    """
    # Whitespace-normalize so sentence splitting isn't confused by paragraph breaks.
    flat = re.sub(r"\s+", " ", text).strip()
    sentences = [s for s in _SENTENCE_RE.split(flat) if s]
    if not sentences:
        return [text]

    sub_chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = current + [sentence]
        if current and _count_tokens(" ".join(candidate)) > MAX_TOKENS:
            sub_chunks.append(" ".join(current))
            # Start the next sub-chunk with one sentence of overlap.
            current = [current[-1], sentence]
        else:
            current.append(sentence)
    if current:
        sub_chunks.append(" ".join(current))
    return sub_chunks


def chunk_text(text: str, dorm_name: str) -> list[dict]:
    """Split one dorm's document into chunks (one chunk per review).

    Returns a list of ``{"text": str, "dorm_name": str, "review_id": str}`` dicts.
    Reviews are split on the ``---`` delimiter. A review longer than MAX_TOKENS is
    sub-split into multiple chunks that share one ``review_id``.
    """
    raw_reviews = _DELIMITER_RE.split(text)
    reviews = [_clean_text(r) for r in raw_reviews]
    reviews = [r for r in reviews if r]  # drop empty segments

    chunks: list[dict] = []
    for index, review in enumerate(reviews, start=1):
        review_id = f"{dorm_name}-{index}"
        if _count_tokens(review) > MAX_TOKENS:
            for sub in _split_long_review(review):
                chunks.append(
                    {"text": sub, "dorm_name": dorm_name, "review_id": review_id}
                )
        else:
            chunks.append(
                {"text": review, "dorm_name": dorm_name, "review_id": review_id}
            )
    return chunks


def build_chunks(documents_dir: os.PathLike | str = DOCUMENTS_DIR) -> list[dict]:
    """Load all documents and chunk them into a single flat list of chunk dicts."""
    chunks: list[dict] = []
    for doc in load_documents(documents_dir):
        chunks.extend(chunk_text(doc["text"], doc["dorm_name"]))
    return chunks


# --------------------------------------------------------------------------- #
# Verification / demo (Milestone 3 verification steps from planning.md)
# --------------------------------------------------------------------------- #


def _verify_and_report(documents_dir: os.PathLike | str = DOCUMENTS_DIR) -> None:
    docs = load_documents(documents_dir)
    known_dorms = {d["dorm_name"] for d in docs}

    total_reviews = 0
    chunks: list[dict] = []
    per_dorm: dict[str, dict] = {}
    for doc in docs:
        n_reviews = len([r for r in _DELIMITER_RE.split(doc["text"]) if r.strip()])
        doc_chunks = chunk_text(doc["text"], doc["dorm_name"])
        total_reviews += n_reviews
        chunks.extend(doc_chunks)
        per_dorm[doc["dorm_name"]] = {
            "reviews": n_reviews,
            "chunks": len(doc_chunks),
        }

    tok = _get_tokenizer()
    print("=" * 70)
    print(f"Tokenizer: {'all-MiniLM-L6-v2 (exact)' if tok else 'word heuristic (fallback)'}")
    print(f"MAX_TOKENS guardrail: {MAX_TOKENS}")
    print("=" * 70)
    print(f"{'dorm':<14}{'reviews':>9}{'chunks':>9}")
    for dorm in sorted(per_dorm):
        info = per_dorm[dorm]
        flag = "  <- sub-split" if info["chunks"] > info["reviews"] else ""
        print(f"{dorm:<14}{info['reviews']:>9}{info['chunks']:>9}{flag}")
    print("-" * 70)
    print(f"{'TOTAL':<14}{total_reviews:>9}{len(chunks):>9}")
    print("=" * 70)

    # (a) chunk count == reviews + sub-chunks from long reviews
    sub_chunk_extra = len(chunks) - total_reviews
    print(f"\n(a) {total_reviews} reviews -> {len(chunks)} chunks "
          f"({sub_chunk_extra} extra from long-review sub-splits)")

    # (b) sub-chunks of the same review share a review_id
    from collections import Counter

    rid_counts = Counter(c["review_id"] for c in chunks)
    shared = {rid: n for rid, n in rid_counts.items() if n > 1}
    print(f"(b) review_ids spanning multiple chunks (sub-split reviews): "
          f"{shared or 'none'}")

    # (c) every chunk carries a correct dorm_name
    bad = [c for c in chunks if c["dorm_name"] not in known_dorms]
    print(f"(c) chunks with unknown dorm_name: {len(bad)}")

    # No chunk should exceed the token ceiling.
    over = [c for c in chunks if _count_tokens(c["text"]) > MAX_TOKENS]
    print(f"    chunks exceeding {MAX_TOKENS} tokens after chunking: {len(over)}")

    assert not bad, "Some chunks have an unknown dorm_name"
    assert len(chunks) >= total_reviews, "Fewer chunks than reviews"

    print("\n" + "=" * 70)
    print("CHUNK PREVIEW")
    print("=" * 70)
    for i, c in enumerate(chunks):
        preview = c["text"].replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        print(f"[{i:>2}] {c['review_id']:<14} ({_count_tokens(c['text']):>3} tok) {preview}")


if __name__ == "__main__":
    _verify_and_report()
