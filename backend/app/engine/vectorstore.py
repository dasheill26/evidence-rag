"""
vectorstore.py

Thin wrapper around ChromaDB - a real, production-used vector database
(HNSW indexing under the hood), not a hand-rolled cosine-similarity
loop over a Python list. That distinction matters: an earlier project
in this portfolio (Face Recognition Studio) implemented similarity
search manually for a small, bounded gallery of known faces, which was
a reasonable choice there but doesn't scale the way a real indexed
vector database does - this project uses the real thing specifically
to close that gap.

Persistent, on-disk storage (not in-memory-only) so a deployed
instance's ingested documents survive a restart.
"""

import os
import uuid
import chromadb

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_store")
COLLECTION_NAME = "documents"

_client = None
_client_db_dir = None  # tracks which DB_DIR the cached client was created for


def get_client():
    """Real bug found and fixed, in two parts: this used to create a
    brand-new chromadb.PersistentClient on every single call, which
    alone produced 'attempt to write a readonly database' errors.
    Caching a single client instance (below) fixed that. But test
    isolation via deleting and recreating the on-disk directory between
    tests turned out to be a second, deeper problem: creating a second
    PersistentClient at the same path within one process fails the same
    way, even after dropping the Python reference to the first client -
    confirmed by reproducing it directly outside pytest, not assumed
    from the pytest failure alone. ChromaDB's PersistentClient is
    designed to be created once per process. The actual fix for test
    isolation was switching to a uniquely-named collection per test
    (see tests/test_pipeline.py) rather than fighting this constraint -
    one client, many logically-separate collections, which is the
    library's own intended usage pattern."""
    global _client, _client_db_dir
    os.makedirs(DB_DIR, exist_ok=True)
    if _client is None or _client_db_dir != DB_DIR:
        _client = chromadb.PersistentClient(path=DB_DIR)
        _client_db_dir = DB_DIR
    return _client


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def add_chunks(chunks, embeddings: list[list[float]], doc_id: str):
    """chunks: list of chunking.Chunk objects. embeddings: parallel list
    of embedding vectors, one per chunk - computed by the caller (not
    inside this module), so this module stays a pure storage layer.

    Real bug found and fixed: chunk storage IDs used to be
    f"{doc_id}::{chunk_index}" - not actually unique if two separate
    documents ever get ingested under the same doc_id, since each
    document's chunks are independently 0-indexed. Confirmed directly:
    ingesting a second document under a reused doc_id silently
    overwrote the first document's chunk-0 entry in ChromaDB - the
    second document appeared to have been "added" (no error) but its
    predecessor's data was gone. Fixed with a genuinely unique ID per
    chunk (doc_id plus a random suffix) regardless of what doc_id the
    caller supplies - chunk_index is kept as metadata for ordering, not
    as part of the identity key."""
    if not chunks:
        return
    collection = get_collection()
    ids = [f"{doc_id}::{uuid.uuid4().hex[:12]}" for _ in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"source": c.source, "chunk_index": c.chunk_index, "doc_id": doc_id} for c in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query(query_embedding: list[float], n_results: int = 4, doc_id: str = None) -> list[dict]:
    """Returns retrieved chunks sorted best-match-first, each with its
    text, source metadata, and distance score - the distance is exposed
    to the caller rather than hidden, since the evaluation harness and
    the UI both need to show how confident a retrieval actually was,
    not just present results as uniformly authoritative."""
    collection = get_collection()
    where = {"doc_id": doc_id} if doc_id else None
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where)

    if not results["ids"] or not results["ids"][0]:
        return []

    return [
        {
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": round(float(results["distances"][0][i]), 4),
        }
        for i in range(len(results["ids"][0]))
    ]


def list_documents() -> list[dict]:
    """Distinct doc_ids currently stored, with chunk counts - lets the
    UI show what's already ingested without needing a separate metadata
    store."""
    collection = get_collection()
    all_data = collection.get(include=["metadatas"])
    doc_counts = {}
    for meta in all_data["metadatas"]:
        doc_id = meta["doc_id"]
        doc_counts[doc_id] = doc_counts.get(doc_id, {"source": meta["source"], "chunks": 0})
        doc_counts[doc_id]["chunks"] += 1
    return [{"doc_id": k, **v} for k, v in doc_counts.items()]


def delete_document(doc_id: str):
    collection = get_collection()
    collection.delete(where={"doc_id": doc_id})
