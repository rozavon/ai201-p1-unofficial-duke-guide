"""Milestone 4 (part 2) — Retrieval for the Unofficial Duke Dorm Guide.

Implements the Retrieval stage from planning.md:
  * Embed the query with the SAME model used at index time (all-MiniLM-L6-v2).
  * Optionally scope to one dorm via metadata filtering (the `dorm_name` field), per
    the Retrieval Approach section.
  * Return the top-k most similar chunks with their source info + distance score.

This is read-only over the Chroma store built by embed.embed_and_store(). Generation
(Milestone 5) consumes retrieve()'s output.
"""

from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from embed import CHROMA_DIR, COLLECTION_NAME, MODEL_NAME

# k=5 (to be tuned by trial and error).
DEFAULT_TOP_K = 5

# Load the embedding model once at import time — the query side must use the same
# model as the index side, so it is sourced from embed.MODEL_NAME.
_model = SentenceTransformer(MODEL_NAME)

# Open the persisted collection once and reuse it across queries.
_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_collection(name=COLLECTION_NAME)


def retrieve(query: str, dorm: str | None = None, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return the top-k chunks most similar to ``query``.

    If ``dorm`` is given, results are restricted to that dorm via a metadata filter
    (``where={"dorm_name": dorm}``) BEFORE semantic ranking, so the k results are the
    closest matches *within that dorm* rather than the closest overall.

    Each result is a dict: ``{text, dorm_name, review_id, position, distance}``.
    ``distance`` is cosine distance (0 = identical direction; lower is more similar).
    """
    query_embedding = _model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).tolist()

    where: chromadb.Where | None = {"dorm_name": dorm} if dorm else None
    result = _collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns one list per query; we only sent one query, so index [0].
    # query() types these fields as optional, so guard before subscripting.
    documents, metadatas, distances = (
        result["documents"],
        result["metadatas"],
        result["distances"],
    )
    assert (
        documents is not None and metadatas is not None and distances is not None
    ), "query() did not return the requested documents/metadatas/distances"
    documents = documents[0]
    metadatas = metadatas[0]
    distances = distances[0]

    return [
        {
            "text": doc,
            "dorm_name": md["dorm_name"],
            "review_id": md["review_id"],
            "position": md["position"],
            "distance": dist,
        }
        for doc, md, dist in zip(documents, metadatas, distances)
    ]


def list_dorms() -> list[str]:
    """Return the distinct ``dorm_name`` values present in the index, sorted.

    Generation (Milestone 5) uses this to detect which dorm(s) a question names so it
    can scope retrieval — and, for multi-dorm comparisons, retrieve per dorm.
    """
    stored = _collection.get(include=["metadatas"])
    metadatas = stored["metadatas"] or []
    return sorted({str(md["dorm_name"]) for md in metadatas})


def _print_results(query: str, results: list[dict]) -> None:
    print("=" * 78)
    print(f"QUERY: {query}")
    print("=" * 78)
    for rank, r in enumerate(results, start=1):
        text = " ".join(r["text"].split())  # flatten newlines for readable preview
        if len(text) > 280:
            text = text[:277] + "..."
        print(
            f"[{rank}] {r['review_id']} (pos {r['position']})  "
            f"distance={r['distance']:.3f}"
        )
        print(f"    {text}")
    print()


# A few of the planning.md Evaluation Plan questions, used to sanity-check retrieval.
_EVAL_QUERIES = [
    ("What are the room sizes like in Randolph?", None),
    ("Does Basset have a mold problem?", None),
    ("How are the bathrooms in Wilson?", None),
]


def _verify_and_report() -> None:
    for query, dorm in _EVAL_QUERIES:
        _print_results(query, retrieve(query, dorm=dorm))

    # Also demonstrate the dorm-scoped path: same query, restricted to one dorm.
    print("### dorm-filtered example (dorm='wilson') ###")
    _print_results("How are the bathrooms?", retrieve("How are the bathrooms?", dorm="wilson"))


if __name__ == "__main__":
    _verify_and_report()
