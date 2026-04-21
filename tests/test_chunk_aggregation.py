"""Unit tests for the chunk-hit aggregator (issue #3)."""

from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp.semantic_search import (
    _aggregate_chunk_hits,
    _check_chunking_compatibility,
    _resolve_chunking_params,
)


def _hit(doc_id: str, parent: str, chunk_index: int, document: str, distance: float):
    """Build a flattened-from-Chroma per-hit dict matching the aggregator's input shape."""
    return {
        "id": doc_id,
        "distance": distance,
        "document": document,
        "metadata": {
            "parent_item_key": parent,
            "chunk_index": chunk_index,
            "chunk_total": 3,
        },
    }


def test_one_result_per_parent():
    """5 hits from 2 parents → 2 results, one per parent."""
    hits = [
        _hit("A:c000", "A", 0, "abs A", 0.10),
        _hit("B:c002", "B", 2, "body B hit", 0.12),
        _hit("A:c001", "A", 1, "intro A", 0.20),
        _hit("A:c002", "A", 2, "body A", 0.25),
        _hit("B:c000", "B", 0, "abs B", 0.40),
    ]
    results = _aggregate_chunk_hits(hits, limit=10)
    assert len(results) == 2
    keys = [r["metadata"]["parent_item_key"] for r in results]
    assert keys == ["A", "B"]  # best-first preserved


def test_best_chunk_wins():
    """The best-scoring chunk's text appears as matched_chunk_text for its paper."""
    # Chroma returns best-first. Paper A's best chunk is c002 with text
    # "body of A" — aggregator should surface that, not the later abstract.
    hits = [
        _hit("A:c002", "A", 2, "body of A", 0.05),  # best for A
        _hit("A:c000", "A", 0, "abstract of A", 0.50),  # worse
    ]
    out = _aggregate_chunk_hits(hits, limit=10)
    assert len(out) == 1
    assert out[0]["matched_chunk_text"] == "body of A"
    assert out[0]["matched_chunk_index"] == 2


def test_limit_respected():
    """limit=10 on 100 distinct parents → 10 results."""
    hits = [_hit(f"P{i}:c000", f"P{i}", 0, f"doc {i}", 0.1 + i * 0.001) for i in range(100)]
    out = _aggregate_chunk_hits(hits, limit=10)
    assert len(out) == 10
    # Must be the first 10 (best-first preserved)
    assert [r["metadata"]["parent_item_key"] for r in out] == [f"P{i}" for i in range(10)]


def test_fallback_key_from_id_suffix():
    """Rows missing parent_item_key (e.g. legacy) fall back to stripping ':cNNN' from id."""
    h = {
        "id": "LEGACYKEY:c007",
        "distance": 0.1,
        "document": "legacy chunk text",
        "metadata": {"chunk_index": 7},  # no parent_item_key
    }
    out = _aggregate_chunk_hits([h], limit=10)
    assert len(out) == 1
    # Deduplicated by the stripped id, i.e. "LEGACYKEY".
    h2 = dict(h)
    h2["id"] = "LEGACYKEY:c008"
    out2 = _aggregate_chunk_hits([h, h2], limit=10)
    assert len(out2) == 1  # same parent after stripping


class _FakeEF:
    def __init__(self, max_tokens):
        self.embedding_max_tokens = max_tokens


def test_resolve_missing_params_points_to_docs():
    """mode='recursive' without chunk sizes raises with a doc pointer."""
    with pytest.raises(ValueError, match="Choosing chunk_size"):
        _resolve_chunking_params({}, _FakeEF(8192))
    with pytest.raises(ValueError, match="chunk_size_tokens"):
        _resolve_chunking_params({"chunk_size_tokens": 100}, _FakeEF(8192))


def test_resolve_rejects_bad_types_and_ranges():
    ef = _FakeEF(8192)
    with pytest.raises(ValueError, match="positive int"):
        _resolve_chunking_params(
            {"chunk_size_tokens": 0, "chunk_overlap_tokens": 0}, ef
        )
    with pytest.raises(ValueError, match="positive int"):
        _resolve_chunking_params(
            {"chunk_size_tokens": -5, "chunk_overlap_tokens": 0}, ef
        )
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_chunking_params(
            {"chunk_size_tokens": 100, "chunk_overlap_tokens": -1}, ef
        )
    with pytest.raises(ValueError, match="must be <"):
        _resolve_chunking_params(
            {"chunk_size_tokens": 100, "chunk_overlap_tokens": 100}, ef
        )
    with pytest.raises(ValueError, match="positive int"):
        # Bool is a subclass of int in Python — must still be rejected.
        _resolve_chunking_params(
            {"chunk_size_tokens": True, "chunk_overlap_tokens": 0}, ef
        )


def test_resolve_clamps_over_ceiling(caplog):
    """chunk_size > 0.6 × embedder max is clamped with a warning, not raised."""
    import logging

    # Ceiling = int(0.6 * 256) = 153
    ef = _FakeEF(256)
    with caplog.at_level(logging.WARNING, logger="zotero_mcp.semantic_search"):
        cs, ov = _resolve_chunking_params(
            {"chunk_size_tokens": 1000, "chunk_overlap_tokens": 50}, ef
        )
    assert cs == 153
    assert ov == 50  # still under clamped cs
    assert any("Clamping to 153" in r.message for r in caplog.records)


def test_resolve_clamps_and_shrinks_overlap_if_needed(caplog):
    """When clamping makes overlap >= cs, overlap is reduced to cs // 8."""
    import logging

    ef = _FakeEF(256)  # ceiling = 153
    # overlap=500 is invalid regardless, so we probe the clamping path with
    # a pre-clamp-valid pair where overlap happens to outgrow the new cs.
    with caplog.at_level(logging.WARNING, logger="zotero_mcp.semantic_search"):
        cs, ov = _resolve_chunking_params(
            {"chunk_size_tokens": 1000, "chunk_overlap_tokens": 200}, ef
        )
    assert cs == 153
    assert ov == cs // 8 == 19


def test_check_compatibility_noop_on_empty_metadata():
    """First-ever indexing (no zmcp keys) must not raise."""
    _check_chunking_compatibility(None, {"mode": "recursive", "chunk_size_tokens": 100})
    _check_chunking_compatibility({}, {"mode": "recursive"})
    _check_chunking_compatibility(
        {"unrelated_key": "x"}, {"mode": "recursive", "chunk_size_tokens": 100}
    )


def test_check_compatibility_detects_mode_switch():
    md = {"zmcp_chunking_mode": "none"}
    with pytest.raises(SystemExit) as ei:
        _check_chunking_compatibility(md, {"mode": "recursive"})
    msg = str(ei.value)
    assert "force-rebuild" in msg
    assert "'none'" in msg and "'recursive'" in msg


def test_check_compatibility_detects_chunk_size_change():
    md = {
        "zmcp_chunking_mode": "recursive",
        "zmcp_chunking_chunk_size_tokens": 800,
        "zmcp_chunking_chunk_overlap_tokens": 80,
    }
    with pytest.raises(SystemExit) as ei:
        _check_chunking_compatibility(md, {
            "mode": "recursive",
            "chunk_size_tokens": 400,
            "chunk_overlap_tokens": 80,
        })
    assert "chunk_size_tokens" in str(ei.value)
    assert "force-rebuild" in str(ei.value)


def test_stable_insertion_order_on_tie():
    """Two chunks with identical score: first-seen wins (stable order)."""
    hits = [
        _hit("A:c001", "A", 1, "first", 0.1),
        _hit("A:c002", "A", 2, "second", 0.1),  # same distance
    ]
    out = _aggregate_chunk_hits(hits, limit=10)
    assert len(out) == 1
    assert out[0]["matched_chunk_index"] == 1  # first seen wins
