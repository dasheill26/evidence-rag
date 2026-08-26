"""
rag_pipeline.py

Ties chunking -> embedding -> vector storage -> retrieval -> generation
into the two operations a caller actually needs: ingest a document, and
answer a question against what's been ingested.

Every answer returns the retrieved chunks alongside it, with their
distance scores - never just a bare answer string. That's a deliberate
design choice, not an afterthought: a RAG system that hides what it
retrieved is much harder to trust or debug, and it's the same "show the
real numbers, not just the polished output" principle the rest of this
portfolio is built around (walk-forward backtest results shown next to
every forecast, calibration scores shown even when they're bad).
"""

import os
import time
from dataclasses import dataclass, field

import anthropic

from .chunking import chunk_text
from .embeddings import embed_texts, embed_query
from . import vectorstore

CLAUDE_MODEL = "claude-sonnet-4-5"
MAX_CONTEXT_CHARS = 6000  # keeps the prompt bounded regardless of how many chunks are retrieved


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float


@dataclass
class RagAnswer:
    question: str
    answer: str
    retrieved_chunks: list
    model: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    estimated_cost_gbp: float


# Claude Sonnet pricing (per million tokens) - used only to produce an
# honest cost estimate alongside each answer, the same cost-accounting
# principle as the Lead Reconciliation Agent project.
INPUT_COST_PER_M = 2.35   # approx GBP per 1M input tokens
OUTPUT_COST_PER_M = 11.75  # approx GBP per 1M output tokens


def ingest_document(text: str, doc_id: str, source: str) -> int:
    """Chunks, embeds, and stores a document. Returns the number of
    chunks created."""
    chunks = chunk_text(text, source=source)
    if not chunks:
        return 0
    embeddings = embed_texts([c.text for c in chunks])
    vectorstore.add_chunks(chunks, embeddings, doc_id=doc_id)
    return len(chunks)


def _build_prompt(question: str, chunks: list) -> str:
    context_parts = []
    total_len = 0
    for c in chunks:
        piece = f"[Source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}"
        if total_len + len(piece) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(piece)
        total_len += len(piece)

    context = "\n\n---\n\n".join(context_parts)
    return (
        "Answer the question using ONLY the context below. If the context doesn't contain "
        "enough information to answer, say so explicitly rather than guessing or using outside "
        "knowledge.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def _simulated_answer(question: str, chunks: list) -> tuple[str, int, int]:
    """No funded Anthropic API key configured - same situation as an
    earlier project in this portfolio (Lead Reconciliation Agent), and
    the same honest fallback: a deterministic, clearly-labeled simulated
    response rather than the app simply not working. Genuinely extracts
    a relevant sentence from the top retrieved chunk rather than a fixed
    placeholder string, so the retrieval half of the pipeline is still
    demonstrated honestly even when the generation half is simulated."""
    top_chunk = chunks[0]["text"] if chunks else ""
    snippet = top_chunk[:220].rsplit(" ", 1)[0] if top_chunk else "no relevant context found"
    answer = (
        f"[SIMULATED - no funded Anthropic API key configured] Based on the top-retrieved chunk "
        f"from '{chunks[0]['source'] if chunks else 'unknown'}': \"{snippet}...\" "
        f"A real deployment with a funded API key would send this context to Claude for a proper "
        f"generated answer instead of this extraction."
    )
    # Rough token estimates for the simulated cost figures - not exact,
    # clearly labeled as estimated in the response, same convention as
    # the Lead Reconciliation Agent project's simulated mode.
    prompt_len = sum(len(c["text"]) for c in chunks) + len(question)
    return answer, prompt_len // 4, len(answer) // 4


def answer_question(question: str, doc_id: str = None, n_chunks: int = 4, api_key: str = None) -> RagAnswer:
    query_vec = embed_query(question)
    raw_chunks = vectorstore.query(query_vec, n_results=n_chunks, doc_id=doc_id)
    retrieved = [RetrievedChunk(**c) for c in raw_chunks]

    if not raw_chunks:
        return RagAnswer(
            question=question,
            answer="No documents have been ingested yet, or none matched this query closely enough to answer from.",
            retrieved_chunks=[], model=CLAUDE_MODEL, latency_seconds=0.0,
            input_tokens=0, output_tokens=0, estimated_cost_gbp=0.0,
        )

    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    t0 = time.time()

    if not resolved_key:
        answer_text, input_tokens, output_tokens = _simulated_answer(question, raw_chunks)
        latency = time.time() - t0
        cost = 0.0
    else:
        prompt = _build_prompt(question, raw_chunks)
        try:
            client = anthropic.Anthropic(api_key=resolved_key)
            response = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            latency = time.time() - t0
            answer_text = "".join(block.text for block in response.content if hasattr(block, "text"))
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens / 1_000_000 * INPUT_COST_PER_M) + (output_tokens / 1_000_000 * OUTPUT_COST_PER_M)
        except Exception as e:
            # A configured-but-invalid/unfunded key (confirmed as a real
            # scenario on an earlier project: a key can authenticate -
            # 401 vs 403 - while still having $0 usable credit) falls
            # back to the same simulated path rather than crashing.
            answer_text, input_tokens, output_tokens = _simulated_answer(question, raw_chunks)
            answer_text = f"[API call failed: {e}] " + answer_text
            latency = time.time() - t0
            cost = 0.0

    return RagAnswer(
        question=question, answer=answer_text, retrieved_chunks=retrieved,
        model=CLAUDE_MODEL, latency_seconds=round(latency, 2),
        input_tokens=input_tokens, output_tokens=output_tokens,
        estimated_cost_gbp=round(cost, 6),
    )
