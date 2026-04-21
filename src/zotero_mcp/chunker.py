"""Dependency-free recursive text splitter for semantic-search chunking.

Chunk size is measured in estimated tokens using a fixed char-per-token
ratio (`_TOKENS_PER_CHAR = 1/3`). This deliberately avoids pulling in a
tokenizer dependency — accuracy is not a goal here; picking chunk
boundaries that honour paragraph/sentence structure is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKENS_PER_CHAR = 1 / 3
# Ordered coarsest → finest. The splitter tries each in turn and stops
# at the first one that yields a boundary in the back half of the window.
_SEPARATORS: list[str] = [r"\n\s*\n", r"\n", r"(?<=[.!?])\s+", r"\s+"]


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int


def recursive_split(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Split `text` into overlapping chunks.

    Splits on paragraph > line > sentence > whitespace > character.
    Returns [] for empty input. Raises ValueError for invalid params.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be > 0")
    if not (0 <= overlap_tokens < chunk_size_tokens):
        raise ValueError("overlap_tokens must be in [0, chunk_size_tokens)")
    if not text or not text.strip():
        return []

    max_chars = int(chunk_size_tokens / _TOKENS_PER_CHAR)
    overlap_chars = int(overlap_tokens / _TOKENS_PER_CHAR)

    chunks: list[Chunk] = []
    i, n = 0, len(text)
    while i < n:
        end_target = min(i + max_chars, n)
        if end_target >= n:
            piece = text[i:n]
            if piece.strip():
                chunks.append(Chunk(piece, len(chunks)))
            break
        # Find the latest separator match in the back half of the window.
        # Prefer coarser separators; fall through to finer ones when no
        # coarse boundary lies in the back half.
        best_end: int | None = None
        lower = i + max_chars // 2
        for sep in _SEPARATORS:
            matches = [
                m.end()
                for m in re.finditer(sep, text[i:end_target])
                if i + m.end() >= lower
            ]
            if matches:
                best_end = i + matches[-1]
                break
        if best_end is None:
            best_end = end_target  # character fallback
        piece = text[i:best_end]
        if piece.strip():
            chunks.append(Chunk(piece, len(chunks)))
        i = max(best_end - overlap_chars, i + 1)
    return chunks
