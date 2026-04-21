"""End-to-end chunking integration tests (issue #3).

Uses a real (tmp-dir-backed) ChromaDB collection and the default
all-MiniLM-L6-v2 embedder. Chunk sizes are chosen below the MiniLM
ceiling (153 tokens = 60% of the 256-token context) so no clamping
happens — otherwise the safeguard tests that compare stored vs.
requested chunk_size would compare clamped-equal values and spuriously
pass.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb", reason="semantic extra not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeZot:
    """Minimal pyzotero stub — just returns an echo dict for item(key)."""

    def item(self, key):
        return {"key": key, "version": 1, "data": {"title": f"T_{key}"}}

    def items(self, **_kwargs):
        return []


def _make_item(key: str, title: str, abstract: str, fulltext: str = ""):
    return {
        "key": key,
        "data": {
            "key": key,
            "itemType": "journalArticle",
            "title": title,
            "abstractNote": abstract,
            "creators": [],
            "fulltext": fulltext,
        },
    }


def _write_config(path: Path, chunking: dict | None):
    cfg: dict = {}
    if chunking is not None:
        cfg = {"semantic_search": {"indexing": {"chunking": chunking}}}
    path.write_text(json.dumps(cfg))


def _make_search(tmp_path: Path, monkeypatch, *, chunking, collection_name: str, items):
    """Build a ZoteroSemanticSearch backed by a real ChromaClient in tmp_path.

    `items` is stored in a closure via monkeypatch so that update_database
    short-circuits `_get_items_from_source` without needing a Zotero backend.
    """
    from zotero_mcp import semantic_search
    from zotero_mcp.chroma_client import ChromaClient

    cfg_path = tmp_path / "config.json"
    _write_config(cfg_path, chunking)

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: _FakeZot())
    client = ChromaClient(
        collection_name=collection_name,
        persist_directory=str(tmp_path / "chroma"),
        embedding_model="default",
    )
    search = semantic_search.ZoteroSemanticSearch(
        chroma_client=client, config_path=str(cfg_path)
    )
    monkeypatch.setattr(search, "_get_items_from_source", lambda **kw: list(items))
    monkeypatch.setattr(search, "_save_update_config", lambda: None)
    return search


def _unique_collection_name() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Tests 12–19 from the spec
# ---------------------------------------------------------------------------

def test_mode_none_unchanged(tmp_path, monkeypatch):
    """mode='none': row IDs equal Zotero item keys; no parent_item_key meta."""
    items = [
        _make_item("AAA111", "Paper One", "First abstract"),
        _make_item("BBB222", "Paper Two", "Second abstract"),
    ]
    search = _make_search(
        tmp_path, monkeypatch,
        chunking={"mode": "none"},
        collection_name=_unique_collection_name(),
        items=items,
    )
    stats = search.update_database()
    assert stats.get("error") is None
    col = search.chroma_client.collection
    data = col.get()
    ids = set(data["ids"])
    assert ids == {"AAA111", "BBB222"}
    for m in data["metadatas"]:
        assert "parent_item_key" not in (m or {})
        assert "chunk_index" not in (m or {})


def test_mode_recursive_produces_chunks(tmp_path, monkeypatch):
    """mode='recursive': row IDs carry ':cNNN' suffix, metadata links back."""
    # 600-word body → plenty of chunks at chunk_size_tokens=60.
    body = ("This sentence contains useful content. " * 30 + "\n\n") * 5
    items = [_make_item("PAPER01", "Big Paper", "abstract", fulltext=body)]
    search = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 60,
            "chunk_overlap_tokens": 6,
        },
        collection_name=_unique_collection_name(),
        items=items,
    )
    stats = search.update_database()
    assert stats.get("error") is None
    data = search.chroma_client.collection.get()
    ids = data["ids"]
    metadatas = data["metadatas"]
    assert len(ids) >= 5, f"expected ≥5 chunks, got {len(ids)}"
    for doc_id, m in zip(ids, metadatas):
        assert doc_id.startswith("PAPER01:c"), doc_id
        assert m.get("parent_item_key") == "PAPER01"
        assert isinstance(m.get("chunk_index"), int)
        assert isinstance(m.get("chunk_total"), int)
        assert m["chunk_total"] == len(ids)


def test_search_returns_distinct_papers(tmp_path, monkeypatch):
    """Query returns no duplicate parent_item_key."""
    items = [
        _make_item("P1", "First paper", "abstract about cats",
                   fulltext=("Cats are elegant creatures " * 40 + ".") * 3),
        _make_item("P2", "Second paper", "abstract about dogs",
                   fulltext=("Dogs are loyal companions " * 40 + ".") * 3),
        _make_item("P3", "Third paper", "abstract about birds",
                   fulltext=("Birds fly through the sky " * 40 + ".") * 3),
    ]
    search = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 60,
            "chunk_overlap_tokens": 6,
        },
        collection_name=_unique_collection_name(),
        items=items,
    )
    search.update_database()
    results = search.search("small animals", limit=5)
    keys = [r["item_key"] for r in results["results"]]
    assert len(keys) == len(set(keys)), f"duplicate parent keys: {keys}"


def test_search_surfaces_matched_chunk(tmp_path, monkeypatch):
    """A distinctive body phrase should come back as matched_chunk_text."""
    # Abstract is about cats; body has a unique marker phrase far from the
    # abstract. Search for the marker phrase — the aggregator should surface
    # the chunk that contains it, not the abstract chunk.
    marker = "XENON_FLAMINGO_QUINTESSENTIAL_ZEBRA_MARKER"
    body = (
        ("Cats wander around lazily. " * 30 + "\n\n") * 2
        + f"{marker}. This passage documents the marker phrase explicitly.\n\n"
        + ("More filler about cats sleeping. " * 30 + "\n\n") * 2
    )
    items = [_make_item("PAPER_M", "Cat habits", "cat abstract", fulltext=body)]
    search = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 50,
            "chunk_overlap_tokens": 5,
        },
        collection_name=_unique_collection_name(),
        items=items,
    )
    search.update_database()
    results = search.search(marker, limit=1)
    assert results["results"], "no results returned"
    hit = results["results"][0]
    assert hit["item_key"] == "PAPER_M"
    assert "matched_chunk_text" in hit
    assert "matched_chunk_index" in hit
    # The matched chunk must contain the marker phrase (or at least part of
    # it — MiniLM on char-boundary chunks may split the token).
    assert marker[:20] in hit["matched_chunk_text"], \
        f"marker not in matched chunk: {hit['matched_chunk_text'][:200]!r}"


def test_mode_mismatch_aborts_without_rebuild(tmp_path, monkeypatch):
    """Switching mode none→recursive without --force-rebuild aborts."""
    coll = _unique_collection_name()
    # Phase 1 — index with mode=none.
    items = [_make_item("MODE1", "Paper", "abstract")]
    search_none = _make_search(
        tmp_path, monkeypatch,
        chunking={"mode": "none"},
        collection_name=coll,
        items=items,
    )
    search_none.update_database()

    # Phase 2 — new instance, same collection, different mode, no rebuild.
    search_rec = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 60,
            "chunk_overlap_tokens": 6,
        },
        collection_name=coll,
        items=items,
    )
    with pytest.raises(SystemExit) as exc_info:
        search_rec.update_database()
    msg = str(exc_info.value)
    assert "force-rebuild" in msg
    assert "none" in msg and "recursive" in msg


def test_chunk_size_change_aborts_without_rebuild(tmp_path, monkeypatch):
    """Changing chunk_size_tokens without --force-rebuild aborts."""
    coll = _unique_collection_name()
    items = [_make_item("CS1", "Paper", "abstract",
                        fulltext=("filler " * 100))]
    # Phase 1 — index with chunk_size_tokens=80.
    search_a = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 80,
            "chunk_overlap_tokens": 8,
        },
        collection_name=coll,
        items=items,
    )
    search_a.update_database()

    # Phase 2 — same collection, different chunk_size.
    search_b = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 40,
            "chunk_overlap_tokens": 4,
        },
        collection_name=coll,
        items=items,
    )
    with pytest.raises(SystemExit) as exc_info:
        search_b.update_database()
    msg = str(exc_info.value)
    assert "chunk_size_tokens" in msg
    assert "force-rebuild" in msg


def test_chunking_metadata_persisted(tmp_path, monkeypatch):
    """After a successful rebuild, collection metadata carries the zmcp keys."""
    items = [_make_item("MD1", "Paper", "abstract",
                        fulltext=("filler " * 100))]
    search = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 70,
            "chunk_overlap_tokens": 7,
        },
        collection_name=_unique_collection_name(),
        items=items,
    )
    search.update_database()
    md = search.chroma_client.get_collection_metadata()
    assert md.get("zmcp_chunking_mode") == "recursive"
    assert md.get("zmcp_chunking_chunk_size_tokens") == 70
    assert md.get("zmcp_chunking_chunk_overlap_tokens") == 7


def test_first_ever_indexing_no_safeguard_fire(tmp_path, monkeypatch):
    """mode='recursive' on an empty collection — no rebuild flag required."""
    items = [_make_item("FIRST1", "Paper", "abstract",
                        fulltext=("filler " * 100))]
    search = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 60,
            "chunk_overlap_tokens": 6,
        },
        collection_name=_unique_collection_name(),
        items=items,
    )
    # Must NOT raise — the safeguard treats missing zmcp_* keys as first-ever.
    stats = search.update_database()
    assert stats.get("error") is None
    assert search.chroma_client.collection.count() >= 1


# ---------------------------------------------------------------------------
# Bonus edge-case coverage (spec-listed edge cases)
# ---------------------------------------------------------------------------

def test_empty_fulltext_still_produces_one_row(tmp_path, monkeypatch):
    """Paper with nothing but a 3-word title → 1 chunk with chunk_total=1.

    Covers the fallback in `_process_item_batch`: when `recursive_split`
    returns [] (or a single chunk) the item still lands in the index.
    """
    items = [_make_item("EMPTY1", "Tiny", "")]
    search = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 60,
            "chunk_overlap_tokens": 6,
        },
        collection_name=_unique_collection_name(),
        items=items,
    )
    search.update_database()
    data = search.chroma_client.collection.get()
    assert len(data["ids"]) == 1
    m = data["metadatas"][0]
    assert m["parent_item_key"] == "EMPTY1"
    assert m["chunk_index"] == 0
    assert m["chunk_total"] == 1


def test_recursive_without_chunk_sizes_aborts(tmp_path, monkeypatch):
    """mode='recursive' without chunk_size_tokens in config → error at update-db start.

    The specific exception type is caught by update_database and surfaces
    via stats['error']; the CLI turns that into a non-zero exit.
    """
    items = [_make_item("X1", "X", "abs")]
    search = _make_search(
        tmp_path, monkeypatch,
        chunking={"mode": "recursive"},  # no chunk sizes
        collection_name=_unique_collection_name(),
        items=items,
    )
    stats = search.update_database()
    assert stats.get("error"), "expected an error"
    assert "chunk_size_tokens" in stats["error"]
    assert "Choosing chunk_size" in stats["error"]


def test_aggregation_toggle_no_rebuild_needed(tmp_path, monkeypatch):
    """`aggregation` is NOT persisted — tweaking it mid-stream must not trip the safeguard.

    This doubles as a pin against accidentally adding `aggregation` to
    `_ZMCP_KEYS` in the future: if someone did, the second run below would
    fail because the first-run collection has no `aggregation` key to match.
    """
    coll = _unique_collection_name()
    items = [_make_item("AGG1", "P", "a", fulltext=("x " * 100))]
    s1 = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 60,
            "chunk_overlap_tokens": 6,
            "aggregation": "max",
        },
        collection_name=coll,
        items=items,
    )
    s1.update_database()

    s2 = _make_search(
        tmp_path, monkeypatch,
        chunking={
            "mode": "recursive",
            "chunk_size_tokens": 60,
            "chunk_overlap_tokens": 6,
            "aggregation": "rrf",  # toggled — must not raise
        },
        collection_name=coll,
        items=items,
    )
    stats = s2.update_database()
    assert stats.get("error") is None
