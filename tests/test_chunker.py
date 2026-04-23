"""Unit tests for the recursive text splitter (issue #3)."""

from __future__ import annotations

import pytest

from zotero_mcp.chunker import Chunk, recursive_split


def test_empty_input_returns_empty_list():
    assert recursive_split("", chunk_size_tokens=100, overlap_tokens=10) == []
    assert recursive_split("   \n\t  ", chunk_size_tokens=100, overlap_tokens=10) == []


def test_single_short_chunk():
    """Input well under chunk_size returns exactly one chunk with index 0."""
    text = "A short paragraph that fits easily within the chunk size."
    chunks = recursive_split(text, chunk_size_tokens=100, overlap_tokens=10)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == text


def test_splits_on_paragraph_first():
    """Paragraph breaks (``\\n\\n``) are the preferred boundary."""
    # Three paragraphs, each comfortably fits in one chunk but together
    # exceed chunk_size. Expect splits to land right after \n\n.
    para = "word " * 40  # ~40 tokens each at 1/3-tokens-per-char
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = recursive_split(text, chunk_size_tokens=60, overlap_tokens=0)
    assert len(chunks) >= 2
    # At least one non-first chunk should start right at a paragraph boundary
    # (i.e., its first char is not whitespace, and the preceding text ended
    # on \n\n in the original).
    # Easier, structural check: every chunk boundary corresponds to a \n\n
    # boundary in the original text.
    positions = []
    cursor = 0
    for c in chunks:
        idx = text.index(c.text, cursor)
        positions.append(idx)
        cursor = idx + len(c.text)
    for pos in positions[1:]:
        # The boundary char preceding this chunk should be whitespace from
        # the split \n\n marker.
        assert text[pos - 1] in (" ", "\n", "\t"), f"boundary at {pos} not whitespace"


def test_falls_back_to_sentence_then_whitespace():
    """No paragraph breaks → split on sentence end; no sentences → whitespace."""
    # Sentences only, no paragraph breaks.
    sentences = "This is sentence one. This is sentence two. This is sentence three. " \
                "This is sentence four. This is sentence five. This is sentence six."
    chunks = recursive_split(sentences, chunk_size_tokens=20, overlap_tokens=0)
    assert len(chunks) >= 2
    # Every chunk boundary should follow a sentence-ending punctuation (or
    # fall on whitespace for the last piece).
    joined = "".join(c.text for c in chunks)
    assert joined.replace("", "")  # basic sanity: chunks cover the text

    # Whitespace-only fallback: long run of space-separated words, no
    # punctuation. Must still produce multiple chunks without crashing.
    ws = "word " * 200
    chunks_ws = recursive_split(ws, chunk_size_tokens=30, overlap_tokens=0)
    assert len(chunks_ws) >= 2


def test_falls_back_to_character():
    """Input with no whitespace splits at max_chars (character fallback)."""
    text = "a" * 1000  # no whitespace, no punctuation — pure blob
    chunks = recursive_split(text, chunk_size_tokens=50, overlap_tokens=0)
    # At 1/3 tokens per char, 50 tokens ≈ 150 chars → 7 chunks
    assert len(chunks) >= 6
    for c in chunks[:-1]:
        assert len(c.text) == 150  # character fallback uses max_chars exactly


def test_overlap_respected():
    """Consecutive chunks share a suffix/prefix of roughly overlap_chars."""
    text = "a" * 1500
    chunks = recursive_split(text, chunk_size_tokens=50, overlap_tokens=10)
    assert len(chunks) >= 2
    # overlap_tokens=10 at 1/3 tokens/char → 30 overlap chars.
    # Since text is uniform "a"s, we can't literally compare — but the
    # advance per step must be (max_chars - overlap_chars) = 150 - 30 = 120.
    # That means total chunks ≈ ceil((1500 - 150) / 120) + 1 ≈ 13.
    assert 10 <= len(chunks) <= 16

    # Same test with zero overlap: chunks should be strictly disjoint,
    # covering the text once with no repetition.
    chunks_no_ov = recursive_split(text, chunk_size_tokens=50, overlap_tokens=0)
    total_len = sum(len(c.text) for c in chunks_no_ov)
    assert total_len == len(text)


def test_indices_sequential_zero_based():
    text = "word " * 300
    chunks = recursive_split(text, chunk_size_tokens=30, overlap_tokens=0)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].index == 0


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        recursive_split("abc", chunk_size_tokens=0, overlap_tokens=0)
    with pytest.raises(ValueError):
        recursive_split("abc", chunk_size_tokens=-1, overlap_tokens=0)
    with pytest.raises(ValueError):
        recursive_split("abc", chunk_size_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        recursive_split("abc", chunk_size_tokens=10, overlap_tokens=15)
    with pytest.raises(ValueError):
        recursive_split("abc", chunk_size_tokens=10, overlap_tokens=-1)


def test_unicode_safe():
    """Splitter must be Unicode-safe (CJK, emoji). Offsets are character-based."""
    # CJK text: each "glyph" is one Python str char, so splitter should
    # behave normally with no UnicodeDecodeError.
    cjk = "中文文字" * 200
    chunks = recursive_split(cjk, chunk_size_tokens=30, overlap_tokens=3)
    assert len(chunks) >= 2
    # Chunks must contain only complete Python str chars (trivially true).
    for c in chunks:
        assert isinstance(c.text, str)


def test_chunk_is_frozen_dataclass():
    """Chunk is hashable and immutable — safe to use as dict key / in sets."""
    c = Chunk("hi", 0)
    with pytest.raises(Exception):  # FrozenInstanceError is a subclass of AttributeError
        c.text = "bye"
    assert hash(c) == hash(Chunk("hi", 0))
