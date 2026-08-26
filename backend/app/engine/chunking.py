"""
chunking.py

Splits documents into overlapping chunks for embedding and retrieval.

Chunk size and overlap are genuine design decisions with real tradeoffs,
not arbitrary defaults - documented here rather than just picked:
  - Too large: each chunk covers multiple topics, so a query about one
    specific fact retrieves a chunk diluted with irrelevant surrounding
    text, and the embedding itself becomes a blurred average of multiple
    concepts, hurting retrieval precision.
  - Too small: a chunk loses surrounding context a human (or the LLM)
    would need to correctly interpret it - a sentence fragment or a
    table row without its header row means almost nothing alone.
  - Overlap exists specifically to avoid silently splitting a sentence
    or idea exactly at a chunk boundary, where the fact needed to answer
    a query ends up split across two chunks and isn't fully present in
    either one.

Sentence-aware splitting (never cutting mid-sentence) rather than a
fixed character count, since a chunk boundary landing mid-word or
mid-sentence measurably hurts both embedding quality and, if it ever
gets shown to a user as a citation, readability.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int
    source: str
    start_char: int
    end_char: int


def split_into_sentences(text: str) -> list[str]:
    """A pragmatic sentence splitter - not a full NLP sentence tokenizer,
    but handles the common cases (. ! ? followed by whitespace and a
    capital letter or end of text) without adding a heavy dependency for
    something this project's actual documents don't need more rigor for."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Groups sentences into chunks of roughly chunk_size characters,
    with the last `overlap` characters' worth of sentences repeated at
    the start of the next chunk. Never splits a sentence in half."""
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_len = 0
    char_offset = 0
    chunk_start_offset = 0

    def flush(end_offset):
        nonlocal current_sentences, current_len
        if not current_sentences:
            return None
        chunk_text_str = " ".join(current_sentences)
        chunk = Chunk(
            text=chunk_text_str, chunk_index=len(chunks), source=source,
            start_char=chunk_start_offset, end_char=end_offset,
        )
        return chunk

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # +1 for the joining space
        if current_len + sentence_len > chunk_size and current_sentences:
            chunk = flush(char_offset)
            chunks.append(chunk)

            # Build overlap: keep trailing sentences from the just-closed
            # chunk whose combined length is <= `overlap`, so the next
            # chunk starts with real context instead of a hard cut.
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) + 1 > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1
            current_sentences = overlap_sentences
            current_len = overlap_len
            chunk_start_offset = char_offset - overlap_len

        current_sentences.append(sentence)
        current_len += sentence_len
        char_offset += sentence_len

    final_chunk = flush(char_offset)
    if final_chunk:
        chunks.append(final_chunk)

    # Re-index chunk_index correctly (flush() doesn't know its own final position)
    for i, c in enumerate(chunks):
        c.chunk_index = i

    return chunks
