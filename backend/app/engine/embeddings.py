"""
embeddings.py

Wraps fastembed (ONNX-runtime based, built by the Qdrant team) rather
than sentence-transformers - both produce genuine, real semantic
embeddings from the same underlying model families, but
sentence-transformers pulls in the full PyTorch stack as a dependency
(confirmed directly: it wouldn't even install in this project's
development sandbox - ran out of disk space on the PyTorch wheel
alone). fastembed uses ONNX Runtime instead, which is dramatically
lighter, without giving up real, meaningful embeddings.

Model: BAAI/bge-small-en-v1.5, 384 dimensions - a small, fast, genuinely
competitive embedding model (not a toy), widely used in production RAG
systems specifically because of that speed/quality tradeoff.
"""

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

_model = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embedding - fastembed's own batching is more efficient than
    embedding one string at a time, so this takes a list, not a single
    string, even when the caller only has one text to embed."""
    if not texts:
        return []
    model = get_model()
    return [emb.tolist() for emb in model.embed(texts)]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
