"""Milestone 5 (part 1) — Grounded generation for the Unofficial Duke Dorm Guide.

Implements the Generation stage from planning.md:
  * Retrieve the most relevant review chunks for the question (retrieve.retrieve()).
  * Build a prompt that ENFORCES grounding — the model may answer ONLY from the
    retrieved reviews, must hedge by reviewer count, and must say it lacks
    information rather than fall back on general knowledge.
  * Call Groq (llama-3.3-70b-versatile, OpenAI-compatible, free tier).
  * Return ``{"answer": str, "sources": list[str]}`` where ``sources`` is built
    PROGRAMMATICALLY from the retrieved chunks' metadata — never left to the LLM to
    invent — so source attribution is guaranteed, not suggested.

Two grounding safeguards beyond the prompt:
  1. If retrieval returns nothing, we short-circuit to the no-information answer and
     never call the LLM at all (it has no context, so it could only hallucinate).
  2. ``sources`` always reflects the actual chunks we retrieved, and is emptied when
     the model returns the no-information sentinel — so a returned source list can
     only ever name documents the answer was actually grounded in.

Multi-dorm questions (e.g. "Is Blackwell or Trinity quieter?") are detected by name
and retrieved per dorm, so one dorm can't crowd the other out of top-k — the
mitigation for Anticipated Challenge #1 in planning.md.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from groq import Groq

from retrieve import DEFAULT_TOP_K, list_dorms, retrieve

# OpenAI-compatible, free-tier model recommended in the milestone brief.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Exact string the model must emit when the context can't answer the question. We
# detect it to blank out the source list, and it is the only allowed "no answer" form.
NO_INFO = "I don't have enough information on that."

# Per-dorm top-k when a question compares multiple dorms, so each named dorm gets fair
# representation instead of one dorm dominating a single shared top-k.
PER_DORM_K = 3

load_dotenv()

# Fail loudly and early if the key is missing, rather than deep inside an API call.
_api_key = os.environ.get("GROQ_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file at the repo root."
    )
_client = Groq(api_key=_api_key)


SYSTEM_PROMPT = f"""You are the Unofficial Duke Dorm Guide. You answer questions about \
Duke University dorms using ONLY the student reviews provided to you in the CONTEXT.

These rules are absolute and override any other instinct:
1. Use ONLY facts stated in the CONTEXT reviews. You have NO other knowledge about \
Duke, its dorms, or anything else. Do not add, infer, or "fill in" anything that is \
not explicitly in the CONTEXT, even if you think it is true or helpful.
2. If the CONTEXT does not contain enough information to answer the question, reply \
with EXACTLY this sentence and nothing else: "{NO_INFO}"
3. Cite the source of every claim using the bracketed tag shown for each review, e.g. \
"(source: basset-2)". A claim with no CONTEXT support is not allowed.
4. Calibrate your confidence to how many reviewers support a claim. If only one review \
mentions something, say so ("one reviewer mentioned..."); if several agree, say \
"multiple reviewers said...". Never present a single opinion as a general fact.
5. Never invent dorm names, reviewers, quotes, or details. If the question asks about a \
dorm that does not appear in the CONTEXT, use the sentence from rule 2.

Answer concisely in plain prose."""


def _detect_dorms(query: str) -> list[str]:
    """Return the indexed dorm names mentioned in ``query`` (case-insensitive, whole word)."""
    mentioned = []
    for dorm in list_dorms():
        if re.search(rf"\b{re.escape(dorm)}\b", query, flags=re.IGNORECASE):
            mentioned.append(dorm)
    return mentioned


def _retrieve_for_query(query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Retrieve chunks, scoping by dorm so multi-dorm comparisons stay balanced.

    - 0 dorms named  -> ordinary top-k over the whole corpus.
    - 1 dorm named   -> top-k scoped to that dorm.
    - 2+ dorms named -> top PER_DORM_K from EACH dorm, concatenated, so neither dorm
      can crowd the other out of a single shared top-k (planning.md challenge #1).
    """
    dorms = _detect_dorms(query)
    if len(dorms) >= 2:
        results: list[dict] = []
        for dorm in dorms:
            results.extend(retrieve(query, dorm=dorm, k=PER_DORM_K))
        return results
    if len(dorms) == 1:
        return retrieve(query, dorm=dorms[0], k=k)
    return retrieve(query, k=k)


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as a tagged CONTEXT block for the prompt.

    Each review is labelled with its ``review_id`` so the model can cite it and so its
    citations line up with the programmatically built source list.
    """
    blocks = []
    for c in chunks:
        text = " ".join(c["text"].split())  # flatten newlines for a clean prompt
        blocks.append(f"[{c['review_id']}] (dorm: {c['dorm_name']})\n{text}")
    return "\n\n".join(blocks)


def _source_list(chunks: list[dict]) -> list[str]:
    """Build the source list PROGRAMMATICALLY from retrieved chunks' metadata.

    Deduplicated by review_id (sub-chunks of one long review share an id), order
    preserved. This is the guaranteed attribution — independent of what the LLM writes.
    """
    seen: set[str] = set()
    sources: list[str] = []
    for c in chunks:
        rid = str(c["review_id"])
        if rid in seen:
            continue
        seen.add(rid)
        sources.append(f"{c['dorm_name']}.txt — review {rid}")
    return sources


def ask(query: str, k: int = DEFAULT_TOP_K) -> dict:
    """Answer ``query`` from retrieved reviews only.

    Returns ``{"answer": str, "sources": list[str]}``. ``sources`` is derived from the
    retrieved chunks (not the model) and is empty whenever the answer is the
    no-information sentinel, so it never names a document the answer didn't use.
    """
    query = (query or "").strip()
    if not query:
        return {"answer": "Please enter a question about a Duke dorm.", "sources": []}

    chunks = _retrieve_for_query(query, k=k)

    # No context retrieved -> never call the LLM; it could only hallucinate.
    if not chunks:
        return {"answer": NO_INFO, "sources": []}

    context = _format_context(chunks)
    user_message = (
        f"CONTEXT (student reviews):\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the CONTEXT above, following all the rules."
    )

    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,  # low -> stay close to the source text, less embellishment
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()

    # If the model couldn't answer from context, don't attribute any sources to it.
    sources = [] if answer == NO_INFO else _source_list(chunks)
    return {"answer": answer, "sources": sources}


def _verify_and_report() -> None:
    """End-to-end grounding check on a few in-corpus queries plus one off-topic query."""
    queries = [
        "What are the room sizes like in Randolph?",
        "Does Basset have a mold problem?",
        "Is Blackwell or Trinity quieter?",  # multi-dorm comparison path
        "What is the meal plan at the engineering quad cafeteria?",  # off-topic
    ]
    for q in queries:
        result = ask(q)
        print("=" * 78)
        print(f"Q: {q}")
        print("-" * 78)
        print(result["answer"])
        print("\nSources:")
        for s in result["sources"] or ["(none)"]:
            print(f"  • {s}")
        print()


if __name__ == "__main__":
    _verify_and_report()
