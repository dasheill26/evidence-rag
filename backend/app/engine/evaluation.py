"""
evaluation.py

Evaluating LLM outputs in a RAG system means checking two genuinely
different things, and conflating them is a common mistake:
  1. Did retrieval find the right information? (a retrieval problem)
  2. Given what was retrieved, is the generated answer actually
     supported by it, or does it hallucinate beyond the provided
     context? (a generation/faithfulness problem)

A RAG system can fail at either stage independently - good retrieval
with a hallucinating generation step, or bad retrieval that an
otherwise-faithful generation step then answers confidently and wrongly
because the right context was never found. This module evaluates both
separately rather than only checking whether the final answer "sounds
right".

Two faithfulness checks, not one, for a real reason: an LLM-as-judge
check (asking Claude whether an answer is supported by its context) is
the more sophisticated, standard technique - but it needs a funded API
key, same constraint as the generation step itself. A lexical-overlap
heuristic (does the answer's vocabulary actually overlap with the
retrieved context, or does it introduce a lot of content that appears
nowhere in what was retrieved?) is cruder but needs no API call at all,
so it's always available - including in a light/demo deployment with no
configured key. Reporting both, honestly labeled with what each one
actually measures, rather than presenting either as a single ground truth.
"""

import re
import os
from dataclasses import dataclass

import anthropic


@dataclass
class FaithfulnessResult:
    method: str  # 'lexical_overlap' or 'llm_judge'
    score: float  # 0-1
    verdict: str
    explanation: str


@dataclass
class RetrievalEvalResult:
    question: str
    expected_source: str
    retrieved_sources: list
    hit: bool
    rank: int  # 1-indexed position of the correct source in results, or -1 if not found


def _tokenize(text: str) -> set:
    # Decimal numbers matched as a single token (\d+\.\d+) BEFORE the
    # plain-integer alternative - otherwise "50.99" and "99.9" would
    # spuriously share the substring "99" once split on the decimal
    # point, an actual bug found by testing: two genuinely different
    # numbers scored as if they overlapped.
    return set(re.findall(r"[a-z]+|\d+\.\d+|\d+", text.lower()))


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on",
    "and", "or", "but", "it", "this", "that", "as", "for", "with", "at", "by", "from",
    "not", "no", "does", "do", "did", "has", "have", "had", "will", "would", "can", "could",
}


def lexical_faithfulness(answer: str, context_chunks: list) -> FaithfulnessResult:
    """Crude but always-available: what fraction of the answer's
    meaningful (non-stopword) vocabulary actually appears somewhere in
    the retrieved context? A low score suggests the answer is
    introducing content the retrieval never actually supplied - not
    proof of hallucination (paraphrasing is normal and healthy), but a
    real, cheap, always-on signal worth surfacing.

    A genuine, tested limitation, not a hypothetical one: a fully
    fabricated answer that keeps the same generic sentence structure as
    the source ("the system uses X and achieved Y percent accuracy")
    can still score ~0.4 on this metric purely from sharing connector
    words like "uses"/"achieved"/"percent" - even when every specific
    claim (X, Y) is invented. Confirmed directly: a test case with
    completely fabricated technology and numbers scored 0.4, not near
    zero. This is exactly why llm_judge_faithfulness below exists as a
    more rigorous complement, not a redundant alternative - it checks
    actual semantic support, not vocabulary overlap."""
    answer_words = _tokenize(answer) - STOPWORDS
    if not answer_words:
        return FaithfulnessResult("lexical_overlap", 1.0, "n/a", "Answer had no scorable content.")

    context_words = set()
    for c in context_chunks:
        context_words |= _tokenize(c["text"])

    overlapping = answer_words & context_words
    score = len(overlapping) / len(answer_words)

    if score >= 0.7:
        verdict = "well grounded"
    elif score >= 0.4:
        verdict = "partially grounded"
    else:
        verdict = "poorly grounded - answer uses vocabulary largely absent from retrieved context"

    return FaithfulnessResult(
        method="lexical_overlap", score=round(score, 3), verdict=verdict,
        explanation=f"{len(overlapping)}/{len(answer_words)} meaningful words in the answer also appear in retrieved context.",
    )


def llm_judge_faithfulness(question: str, answer: str, context_chunks: list, api_key: str = None) -> FaithfulnessResult:
    """The more rigorous check - asks Claude directly whether the answer
    is actually supported by the context, not just lexically similar to
    it. Needs a funded API key; falls back to a clearly-labeled
    unavailable result rather than a fabricated score when one isn't
    configured, the same honesty principle as the generation step."""
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        return FaithfulnessResult(
            "llm_judge", -1.0, "unavailable",
            "No funded Anthropic API key configured - this check needs a real LLM call to judge faithfulness, unlike the lexical check above.",
        )

    context = "\n\n".join(c["text"] for c in context_chunks)
    judge_prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer given: {answer}\n\n"
        "Is this answer fully supported by the context above? Reply with exactly one line in the "
        "format: VERDICT: <supported|partially_supported|unsupported> | REASON: <one sentence>"
    )
    try:
        client = anthropic.Anthropic(api_key=resolved_key)
        response = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=150,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        verdict_match = re.search(r"VERDICT:\s*(\w+)", text)
        reason_match = re.search(r"REASON:\s*(.+)", text)
        verdict = verdict_match.group(1) if verdict_match else "unknown"
        reason = reason_match.group(1) if reason_match else text
        score = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}.get(verdict, 0.5)
        return FaithfulnessResult("llm_judge", score, verdict, reason)
    except Exception as e:
        return FaithfulnessResult("llm_judge", -1.0, "error", f"Judge call failed: {e}")


def evaluate_retrieval(test_cases: list, doc_id: str = None, n_chunks: int = 4) -> list:
    """test_cases: list of {"question": str, "expected_source": str}.
    Runs actual retrieval for each and checks whether the expected
    source appears in the results, at what rank - a real precision/recall
    style check against known ground truth, not a vibe check."""
    from . import vectorstore
    from .embeddings import embed_query

    results = []
    for case in test_cases:
        query_vec = embed_query(case["question"])
        retrieved = vectorstore.query(query_vec, n_results=n_chunks, doc_id=doc_id)
        sources = [r["source"] for r in retrieved]
        rank = -1
        for i, s in enumerate(sources):
            if s == case["expected_source"]:
                rank = i + 1
                break
        results.append(RetrievalEvalResult(
            question=case["question"], expected_source=case["expected_source"],
            retrieved_sources=sources, hit=rank != -1, rank=rank,
        ))
    return results
