"""Milestone 4 (part 1) — Embedding and vector storage for the Unofficial Duke Dorm Guide.

Implements the Embedding + Vector Store stage from planning.md:
  * Embed each chunk with all-MiniLM-L6-v2 (sentence-transformers) -> 384-dim vector.
  * Persist vectors + text + metadata (dorm_name, review_id) in a local ChromaDB.

The chunks come from ingest.build_chunks() (Milestone 3). Retrieval (retrieve(),
the second half of Milestone 4) and generation (Milestone 5) live elsewhere.

Note on ids: planning.md deliberately lets sub-chunks of one long review SHARE a
review_id, so review_id is not unique. ChromaDB requires a unique id per item, so we
derive a distinct chunk_id per chunk while keeping review_id intact in the metadata.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import build_chunks

# Same embedding model as planning.md's Retrieval Approach (and the one ingest.py's
# token guardrail is sized for). The query side (retrieve()) must use this same model.
MODEL_NAME = "all-MiniLM-L6-v2"

# Persisted Chroma store. Matches the `chroma_db/` entry already in .gitignore.
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "duke_dorm_reviews"


def _assign_chunk_ids(chunks: list[dict]) -> list[str]:
    """Derive a unique, deterministic id per chunk.

    review_id is shared across sub-chunks of one long review, so we append a per-
    review_id sub-index: "basset-1" -> "basset-1-1", "basset-1-2", etc. Deterministic
    ids let re-running this script upsert in place instead of duplicating rows.
    """
    seen: dict[str, int] = defaultdict(int)
    ids: list[str] = []
    for chunk in chunks:
        rid = chunk["review_id"]
        seen[rid] += 1
        ids.append(f"{rid}-{seen[rid]}")
    return ids


def embed_and_store(
    chroma_dir: Path | str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = MODEL_NAME,
) -> chromadb.Collection:
    """Embed every chunk and write it to a persistent ChromaDB collection.

    Each stored item carries the chunk text as the document, its embedding, and
    {dorm_name, review_id} metadata. Uses upsert with deterministic ids so re-running
    rebuilds the index in place rather than creating duplicates.

    Returns the populated Chroma collection.
    """
    chunks = build_chunks()
    if not chunks:
        raise RuntimeError("No chunks produced by ingest.build_chunks() — nothing to embed.")

    ids = _assign_chunk_ids(chunks)
    texts = [c["text"] for c in chunks]

    # Source document (dorm_name) + the chunk's position within that document.
    # review_id keeps the review-level grouping (sub-chunks share it); `position` is a
    # per-dorm 1-based ordinal so attribution can point at an exact chunk later.
    position_in_dorm: dict[str, int] = defaultdict(int)
    metadatas: list[chromadb.Metadata] = []
    for chunk in chunks:
        position_in_dorm[chunk["dorm_name"]] += 1
        metadatas.append(
            {
                "dorm_name": chunk["dorm_name"],
                "review_id": chunk["review_id"],
                "position": position_in_dorm[chunk["dorm_name"]],
            }
        )

    # Embed all chunks with the same model the query side will use.
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()

    # Persistent on-disk store so the index survives between runs / processes.
    client = chromadb.PersistentClient(path=str(chroma_dir))
    # We supply embeddings explicitly, so no embedding_function is attached to the
    # collection. Cosine space matches the normalized MiniLM vectors above.
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection


def _verify_and_report() -> None:
    collection = embed_and_store()
    count = collection.count()

    print("=" * 70)
    print(f"Embedding model:   {MODEL_NAME}")
    print(f"Chroma collection: {COLLECTION_NAME}")
    print(f"Persist directory: {CHROMA_DIR}")
    print("=" * 70)
    print(f"Stored items: {count}")

    # Per-dorm counts, read back from the store to confirm metadata round-tripped.
    stored = collection.get(include=["metadatas"])
    metadatas = stored["metadatas"]
    assert metadatas is not None, "get(include=['metadatas']) returned no metadatas"
    per_dorm: dict[str, int] = defaultdict(int)
    for md in metadatas:
        per_dorm[str(md["dorm_name"])] += 1
    print("-" * 70)
    print(f"{'dorm':<14}{'chunks':>9}")
    for dorm in sorted(per_dorm):
        print(f"{dorm:<14}{per_dorm[dorm]:>9}")
    print("-" * 70)

    # Confirm vectors are the expected 384-dim and the id derivation stayed unique.
    sample = collection.get(ids=[stored["ids"][0]], include=["embeddings"])
    embeddings = sample["embeddings"]
    assert embeddings is not None, "get(include=['embeddings']) returned no embeddings"
    dim = len(embeddings[0])
    print(f"Embedding dimension: {dim} (expected 384)")
    print(f"Unique ids: {len(set(stored['ids']))} / {count} stored")
    assert len(set(stored["ids"])) == count, "Duplicate chunk ids detected"
    assert dim == 384, f"Unexpected embedding dimension: {dim}"
    print("\nOK — chunks embedded and stored in ChromaDB.")


if __name__ == "__main__":
    _verify_and_report()
