"""
Tests for Evidence RAG. Uses mocked embeddings (deterministic random
vectors of the real 384 dimension) rather than the live fastembed model
download, for the same reason documented in embeddings.py: HuggingFace
and GitHub's LFS media server are both outside this development
sandbox's network allowlist. The chunking, vector storage, retrieval
ordering, evaluation, and API contract logic are all tested directly
and for real - only the specific embedding *values* are substituted;
the mechanics around them are not.
"""
import sys
import os
import shutil
import uuid

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CHROMA_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_store")


@pytest.fixture(scope="session", autouse=True)
def clean_chroma_store_once():
    """One-time cleanup at the start of the whole test session, so
    collections from a previous test run don't accumulate indefinitely
    on disk - each individual test still gets its own uniquely-named
    collection via clean_vectorstore below, this just keeps the store
    from growing unboundedly across repeated test runs."""
    shutil.rmtree(CHROMA_TEST_DIR, ignore_errors=True)
    yield


@pytest.fixture(autouse=True)
def clean_vectorstore(monkeypatch):
    """Test isolation via a uniquely-named ChromaDB collection per test,
    not by destroying and recreating the PersistentClient - a real,
    confirmed issue found while building this: creating a second
    PersistentClient at the same on-disk path within one process (even
    after dropping the Python reference to the first one and resetting
    a module-level cache variable) reliably produced 'attempt to write
    a readonly database', reproduced directly outside pytest to confirm
    it wasn't a pytest-specific quirk. ChromaDB's PersistentClient is
    meant to be created once per process and reused; a unique collection
    name per test achieves the same data isolation without fighting
    that constraint - one client, many logically-separate collections,
    which is exactly the intended usage pattern."""
    import uuid
    from app.engine import vectorstore
    unique_name = f"test_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(vectorstore, "COLLECTION_NAME", unique_name)
    yield


# ---------- Chunking ----------

from app.engine.chunking import chunk_text, split_into_sentences


def test_sentence_splitting_handles_common_punctuation():
    sentences = split_into_sentences("This is one. This is two! Is this three? Yes it is.")
    assert len(sentences) == 4


def test_chunks_never_split_a_sentence_in_half():
    text = " ".join(f"This is sentence number {i} with padding words to make it longer." for i in range(30))
    chunks = chunk_text(text, source="test.txt", chunk_size=300, overlap=50)
    for c in chunks:
        assert not c.text.rstrip().endswith(("with", "to", "a", "the", "make"))


def test_consecutive_chunks_share_overlapping_content():
    text = " ".join(f"This is sentence number {i} with padding words to make it longer." for i in range(30))
    chunks = chunk_text(text, source="test.txt", chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    tail_words = set(chunks[0].text[-40:].split())
    head_words = set(chunks[1].text[:100].split())
    assert len(tail_words & head_words) > 0


def test_empty_text_produces_no_chunks():
    assert chunk_text("", source="empty.txt") == []


# ---------- Vector store (mocked embeddings, real ChromaDB) ----------

from app.engine import vectorstore


def _fake_vector(seed_text: str) -> list:
    rng = np.random.RandomState(abs(hash(seed_text)) % (2**32))
    return rng.rand(384).tolist()


def test_exact_match_query_returns_near_zero_distance():
    chunks = chunk_text("A single short document.", source="doc.txt", chunk_size=100)
    vec = _fake_vector(chunks[0].text)
    vectorstore.add_chunks(chunks, [vec], doc_id="d1")
    results = vectorstore.query(vec, n_results=1)
    assert results[0]["distance"] < 0.01


def test_metadata_filtering_does_not_leak_across_documents():
    chunks_a = chunk_text("Content about apples.", source="a.txt", chunk_size=100)
    chunks_b = chunk_text("Content about rockets.", source="b.txt", chunk_size=100)
    vectorstore.add_chunks(chunks_a, [_fake_vector(chunks_a[0].text)], doc_id="doc-a")
    vectorstore.add_chunks(chunks_b, [_fake_vector(chunks_b[0].text)], doc_id="doc-b")
    results = vectorstore.query(_fake_vector(chunks_a[0].text), n_results=5, doc_id="doc-b")
    assert all(r["source"] == "b.txt" for r in results)


def test_delete_document_removes_only_the_targeted_document():
    chunks_a = chunk_text("Content A.", source="a.txt", chunk_size=100)
    chunks_b = chunk_text("Content B.", source="b.txt", chunk_size=100)
    vectorstore.add_chunks(chunks_a, [_fake_vector(chunks_a[0].text)], doc_id="doc-a")
    vectorstore.add_chunks(chunks_b, [_fake_vector(chunks_b[0].text)], doc_id="doc-b")
    vectorstore.delete_document("doc-a")
    docs = vectorstore.list_documents()
    assert len(docs) == 1 and docs[0]["doc_id"] == "doc-b"


def test_documents_sharing_a_doc_id_do_not_overwrite_each_other():
    """Regression test for a real, serious bug found during development:
    chunk storage IDs used to be f"{doc_id}::{chunk_index}" - not
    globally unique if two different documents ever get ingested under
    the same doc_id, since each document's chunks are independently
    0-indexed. Confirmed directly: ingesting a second document under a
    reused doc_id silently overwrote the first document's chunk-0 entry
    with no error. Fixed with a genuinely unique ID per chunk regardless
    of doc_id reuse."""
    chunks_a = chunk_text("Fact about apples.", source="fruit.txt", chunk_size=100)
    chunks_b = chunk_text("Fact about rockets.", source="space.txt", chunk_size=100)
    vec_a = _fake_vector("apples-vector")
    vec_b = _fake_vector("rockets-vector")

    vectorstore.add_chunks(chunks_a, [vec_a], doc_id="shared-id")
    vectorstore.add_chunks(chunks_b, [vec_b], doc_id="shared-id")  # same doc_id, deliberately

    result_a = vectorstore.query(vec_a, n_results=2, doc_id="shared-id")
    result_b = vectorstore.query(vec_b, n_results=2, doc_id="shared-id")
    assert result_a[0]["source"] == "fruit.txt" and result_a[0]["distance"] < 0.01
    assert result_b[0]["source"] == "space.txt" and result_b[0]["distance"] < 0.01


# ---------- Evaluation ----------

from app.engine.evaluation import lexical_faithfulness, llm_judge_faithfulness, _tokenize


def test_decimal_numbers_do_not_spuriously_overlap():
    """Regression test for a real bug: the tokenizer split on decimal
    points, so '50.99' and '99.9' shared the substring '99' even though
    they're completely different numbers."""
    assert _tokenize("50.99") & _tokenize("99.9") == set()


def test_lexical_faithfulness_ranks_grounded_above_hallucinated():
    context = [{"text": "Market Signal Lab uses walk-forward backtesting. It achieved 50.99 percent accuracy on NVDA data."}]
    grounded = lexical_faithfulness("The model uses walk-forward backtesting and achieved 50.99 percent accuracy.", context)
    hallucinated = lexical_faithfulness("The system uses quantum computing and achieved 99.9 percent accuracy with blockchain integration.", context)
    assert grounded.score > hallucinated.score
    assert grounded.verdict == "well grounded"


def test_llm_judge_honestly_reports_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = llm_judge_faithfulness("q", "a", [{"text": "ctx"}])
    assert result.verdict == "unavailable"
    assert result.score == -1.0


def test_llm_judge_parses_a_real_shaped_claude_response(monkeypatch):
    from unittest.mock import patch, MagicMock
    fake_response = MagicMock()
    fake_block = MagicMock()
    fake_block.text = "VERDICT: supported | REASON: Directly restated in the context."
    fake_response.content = [fake_block]
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = fake_response
        result = llm_judge_faithfulness("q", "a", [{"text": "ctx"}], api_key="fake-key")
    assert result.verdict == "supported"
    assert result.score == 1.0


# ---------- Full pipeline ----------

from app.engine.rag_pipeline import ingest_document, answer_question


def test_full_pipeline_ingest_and_answer(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    doc = (
        "Market Signal Lab is a full-stack ML forecasting app. "
        "It uses walk-forward backtesting to avoid training on future data. "
        "On real NVDA data it achieved 50.99 percent accuracy versus a 49.49 percent baseline."
    )
    from unittest.mock import patch
    with patch("app.engine.rag_pipeline.embed_texts", side_effect=lambda texts: [_fake_vector(t) for t in texts]), \
         patch("app.engine.rag_pipeline.embed_query", side_effect=lambda q: _fake_vector(q)):
        n_chunks = ingest_document(doc, doc_id="test-doc", source="test.txt")
        assert n_chunks > 0
        result = answer_question("What backtesting method is used?", doc_id="test-doc")
    assert result.retrieved_chunks
    assert "[SIMULATED" in result.answer  # no API key configured in this test


def test_answer_question_with_no_ingested_documents_does_not_crash():
    from unittest.mock import patch
    with patch("app.engine.rag_pipeline.embed_query", side_effect=lambda q: _fake_vector(q)):
        result = answer_question("anything", doc_id="nonexistent-doc-id")
    assert result.retrieved_chunks == []


# ---------- API layer ----------

from run import app as flask_app


def test_api_healthz():
    client = flask_app.test_client()
    r = client.get("/healthz")
    assert r.status_code == 200


def test_api_upload_missing_content_returns_clean_400():
    client = flask_app.test_client()
    r = client.post("/api/documents", json={})
    assert r.status_code == 400


def test_api_upload_and_list_document(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("app.engine.rag_pipeline.embed_texts", side_effect=lambda texts: [_fake_vector(t) for t in texts]):
        client = flask_app.test_client()
        r = client.post("/api/documents", json={"text": "Some real content to ingest.", "source": "api-test.txt"})
        assert r.status_code == 200
        data = r.get_json()
        assert data["chunks_created"] > 0

        r2 = client.get("/api/documents")
        sources = [d["source"] for d in r2.get_json()["documents"]]
        assert "api-test.txt" in sources


def test_api_ask_missing_question_returns_clean_400():
    client = flask_app.test_client()
    r = client.post("/api/ask", json={})
    assert r.status_code == 400


def test_api_ask_full_flow(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("app.engine.rag_pipeline.embed_texts", side_effect=lambda texts: [_fake_vector(t) for t in texts]), \
         patch("app.engine.rag_pipeline.embed_query", side_effect=lambda q: _fake_vector(q)):
        client = flask_app.test_client()
        client.post("/api/documents", json={"text": "The sky is blue during a clear day.", "source": "sky.txt"})
        r = client.post("/api/ask", json={"question": "What color is the sky?"})
        assert r.status_code == 200
        data = r.get_json()
        assert "answer" in data
        assert "evaluation" in data
        assert data["evaluation"]["lexical_faithfulness"] is not None


if __name__ == "__main__":
    test_sentence_splitting_handles_common_punctuation()
    test_chunks_never_split_a_sentence_in_half()
    test_decimal_numbers_do_not_spuriously_overlap()
    print("Core tests passed (run via pytest for the full suite).")
