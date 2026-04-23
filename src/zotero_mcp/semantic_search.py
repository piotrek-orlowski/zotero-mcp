"""
Semantic search functionality for Zotero MCP.

This module provides semantic search capabilities by integrating ChromaDB
with the existing Zotero client to enable vector-based similarity search
over research libraries.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import logging

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    _tokenizer = None

from pyzotero import zotero

from .chroma_client import ChromaClient, create_chroma_client
from .client import get_zotero_client
from .utils import format_creators, is_local_mode
from .local_db import LocalZoteroReader, get_local_zotero_reader

logger = logging.getLogger(__name__)


from zotero_mcp.utils import suppress_stdout


def _truncate_to_tokens(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to fit within embedding model token limit.

    Uses tiktoken for accurate token counting when available,
    falls back to conservative character-based estimation.
    """
    if _tokenizer is not None:
        tokens = _tokenizer.encode(text, disallowed_special=())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = _tokenizer.decode(tokens)
    else:
        # Fallback: conservative char limit (~1.5 chars/token for non-Latin scripts)
        max_chars = max_tokens * 2
        if len(text) > max_chars:
            text = text[:max_chars]
    return text


# ---------------------------------------------------------------------------
# Chunking support (issue #3)
#
# Persisted keys under the `zmcp_` prefix so they can't collide with any
# internal ChromaDB metadata. Only the three collection-defining params
# (mode + two chunk-size fields) live here. Operational/search-side knobs
# must NOT be persisted — see gotcha #7 in issue #1.
# ---------------------------------------------------------------------------
_ZMCP_KEYS = (
    "zmcp_chunking_mode",
    "zmcp_chunking_chunk_size_tokens",
    "zmcp_chunking_chunk_overlap_tokens",
)


def _resolve_chunking_params(
    chunking_cfg: dict,
    ef,
) -> tuple[int, int]:
    """Resolve and validate chunk_size / chunk_overlap against the active
    embedding function's max_tokens.

    Refuses silent defaults; clamps with a warning on over-large values.
    """
    cs = chunking_cfg.get("chunk_size_tokens")
    ov = chunking_cfg.get("chunk_overlap_tokens")
    if cs is None or ov is None:
        raise ValueError(
            "mode='recursive' requires both chunk_size_tokens and "
            "chunk_overlap_tokens in config. See docs section "
            "'Choosing chunk_size' for per-embedder recommendations."
        )
    if not isinstance(cs, int) or isinstance(cs, bool) or cs <= 0:
        raise ValueError(f"chunk_size_tokens must be a positive int, got {cs!r}")
    if not isinstance(ov, int) or isinstance(ov, bool) or ov < 0:
        raise ValueError(
            f"chunk_overlap_tokens must be a non-negative int, got {ov!r}"
        )
    if ov >= cs:
        raise ValueError(
            f"chunk_overlap_tokens ({ov}) must be < chunk_size_tokens ({cs})"
        )

    ceiling = int(getattr(ef, "embedding_max_tokens", 2048) * 0.6)
    if cs > ceiling:
        logger.warning(
            f"chunk_size_tokens={cs} exceeds safe embedder ceiling "
            f"({ceiling}, = 60% of {getattr(ef, 'embedding_max_tokens', 2048)}). "
            f"Clamping to {ceiling} to avoid per-chunk truncation. "
            f"Consider a smaller chunk_size, or switch to an embedder "
            f"with a larger context window."
        )
        cs = ceiling
        if ov >= cs:
            ov = max(0, cs // 8)
    return cs, ov


def _check_chunking_compatibility(
    collection_metadata: dict | None,
    config: dict,
) -> None:
    """Compare persisted chunking params against requested config.

    Raises SystemExit with a pointer to `--force-rebuild` if anything
    material differs. Missing persisted keys are treated as "no prior
    chunking state" → first-ever indexing proceeds without comparison.
    """
    md = collection_metadata or {}
    if not any(k in md for k in _ZMCP_KEYS):
        return

    stored_mode = md.get("zmcp_chunking_mode", "none")
    requested_mode = config.get("mode", "none")
    if stored_mode != requested_mode:
        raise SystemExit(
            f"Collection was indexed with chunking mode={stored_mode!r}, "
            f"but config says {requested_mode!r}. "
            f"Run `update-db --force-rebuild` to switch."
        )
    if requested_mode == "recursive":
        for key, cfg_key in [
            ("zmcp_chunking_chunk_size_tokens", "chunk_size_tokens"),
            ("zmcp_chunking_chunk_overlap_tokens", "chunk_overlap_tokens"),
        ]:
            stored = md.get(key)
            requested = config.get(cfg_key)
            if stored is not None and stored != requested:
                raise SystemExit(
                    f"Collection was indexed with {cfg_key}={stored}, "
                    f"config says {requested}. Chunk content differs; "
                    f"mixing would produce inconsistent retrieval. "
                    f"Run `update-db --force-rebuild` to switch."
                )


def _aggregate_chunk_hits(hits: list[dict], limit: int) -> list[dict]:
    """Aggregate a best-first list of chunk hits into one hit per paper.

    hits: per-hit dicts with keys {"id", "document", "metadata", ...}
          — typically flattened from ChromaDB's nested-list response.
    Returns up to `limit` hits, each augmented with
    `matched_chunk_text` and `matched_chunk_index`.
    """
    seen: dict[str, dict] = {}
    for h in hits:
        meta = h.get("metadata") or {}
        key = meta.get("parent_item_key")
        if not key:
            # Non-chunked row (or chunked row with no parent key): fall
            # back to stripping the chunk suffix from the id.
            doc_id = h.get("id", "") or ""
            key = doc_id.split(":c")[0] if ":c" in doc_id else doc_id
        if not key or key in seen:
            continue
        h2 = dict(h)
        h2["matched_chunk_text"] = h.get("document", "")
        h2["matched_chunk_index"] = meta.get("chunk_index", 0)
        seen[key] = h2
        if len(seen) >= limit:
            break
    return list(seen.values())


def _chroma_results_to_hits(results: dict) -> list[dict]:
    """Flatten a ChromaDB nested-list query response into per-hit dicts."""
    ids = (results.get("ids") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    hits: list[dict] = []
    for i, doc_id in enumerate(ids):
        hits.append({
            "id": doc_id,
            "distance": distances[i] if i < len(distances) else None,
            "document": documents[i] if i < len(documents) else "",
            "metadata": metadatas[i] if i < len(metadatas) else {},
        })
    return hits


class CrossEncoderReranker:
    """Optional cross-encoder re-ranker for semantic search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Re-rank documents by relevance to query.

        Returns indices of top_k documents in descending relevance order.
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_indices[:top_k]


class ZoteroSemanticSearch:
    """Semantic search interface for Zotero libraries using ChromaDB."""

    def __init__(self,
                 chroma_client: ChromaClient | None = None,
                 config_path: str | None = None,
                 db_path: str | None = None):
        """
        Initialize semantic search.

        Args:
            chroma_client: Optional ChromaClient instance
            config_path: Path to configuration file
            db_path: Optional path to Zotero database (overrides config file)
        """
        self.chroma_client = chroma_client or create_chroma_client(config_path)
        self.zotero_client = get_zotero_client()
        self.config_path = config_path
        self.db_path = db_path  # CLI override for Zotero database path

        # Load update configuration
        self.update_config = self._load_update_config()

        # Reranker (lazy-initialized on first search)
        self._reranker: CrossEncoderReranker | None = None
        self._reranker_config = self._load_reranker_config()

        # Pre-embedding text-extraction cleanup config
        self._extraction_config = self._load_extraction_config()

        # Chunking config (issue #3). Collection-defining params — changing
        # them requires --force-rebuild, enforced by the safeguard in
        # update_database. Default is mode='none' → pipeline is byte-identical
        # to pre-#3 behaviour.
        self._chunking_config = self._load_chunking_config()

    def _load_reranker_config(self) -> dict[str, Any]:
        """Load reranker configuration from file or use defaults."""
        config: dict[str, Any] = {
            "enabled": False,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "candidate_multiplier": 3,
        }
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("reranker", {}))
            except Exception as e:
                logger.warning(f"Error loading reranker config: {e}")
        return config

    def _load_extraction_config(self) -> dict[str, Any]:
        """Load pre-embedding extraction/cleanup config. Defaults-on."""
        config: dict[str, Any] = {
            "strip_boilerplate": True,
            "skip_to_abstract": True,
        }
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("extraction", {}))
            except Exception as e:
                logger.warning(f"Error loading extraction config: {e}")
        return config

    def _load_chunking_config(self) -> dict[str, Any]:
        """Load chunking configuration. Defaults to mode='none' (disabled)."""
        config: dict[str, Any] = {
            "mode": "none",
            "chunk_size_tokens": None,
            "chunk_overlap_tokens": None,
        }
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    chunking = (
                        file_config.get("semantic_search", {})
                        .get("indexing", {})
                        .get("chunking", {})
                        or {}
                    )
                    config.update(chunking)
            except Exception as e:
                logger.warning(f"Error loading chunking config: {e}")
        mode = config.get("mode", "none")
        if mode not in ("none", "recursive"):
            raise ValueError(f"Unknown chunking mode: {mode!r}")
        return config

    def _get_reranker(self) -> CrossEncoderReranker | None:
        """Get the reranker instance, lazily initializing if enabled."""
        if not self._reranker_config.get("enabled", False):
            return None
        if self._reranker is None:
            model = self._reranker_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._reranker = CrossEncoderReranker(model_name=model)
        return self._reranker

    def _load_update_config(self) -> dict[str, Any]:
        """Load update configuration from file or use defaults."""
        config = {
            "auto_update": False,
            "update_frequency": "manual",
            "last_update": None,
            "update_days": 7
        }

        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("update_config", {}))
            except Exception as e:
                logger.warning(f"Error loading update config: {e}")

        return config

    def _save_update_config(self) -> None:
        """Save update configuration to file."""
        if not self.config_path:
            return

        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new one
        full_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    full_config = json.load(f)
            except Exception:
                pass

        # Update semantic search config
        if "semantic_search" not in full_config:
            full_config["semantic_search"] = {}

        full_config["semantic_search"]["update_config"] = self.update_config

        try:
            with open(self.config_path, 'w') as f:
                json.dump(full_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving update config: {e}")

    def _create_document_text(self, item: dict[str, Any]) -> str:
        """
        Create searchable text from a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Combined text for embedding
        """
        data = item.get("data", {})

        # Extract key fields for semantic search
        title = data.get("title", "")
        abstract = data.get("abstractNote", "")

        # Format creators as text
        creators = data.get("creators", [])
        creators_text = format_creators(creators)

        # Additional searchable content
        extra_fields = []

        # Publication details
        if publication := data.get("publicationTitle"):
            extra_fields.append(publication)

        # Tags
        if tags := data.get("tags"):
            tag_text = " ".join([tag.get("tag", "") for tag in tags])
            extra_fields.append(tag_text)

        # Note content (if available)
        if note := data.get("note"):
            # Clean HTML from notes
            import re
            note_text = re.sub(r'<[^>]+>', '', note)
            extra_fields.append(note_text)

        # Combine all text fields
        text_parts = [title, creators_text, abstract] + extra_fields
        return " ".join(filter(None, text_parts))

    def _create_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Create metadata for a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Metadata dictionary for ChromaDB
        """
        data = item.get("data", {})

        metadata = {
            "item_key": item.get("key", ""),
            "item_type": data.get("itemType", ""),
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "date_added": data.get("dateAdded", ""),
            "date_modified": data.get("dateModified", ""),
            "creators": format_creators(data.get("creators", [])),
            "publication": data.get("publicationTitle", ""),
            "url": data.get("url", ""),
            "doi": data.get("DOI", ""),
        }
        # If fulltext was extracted (or attempted), mark it so incremental
        # updates don't keep re-trying items that failed extraction
        if data.get("fulltext"):
            metadata["has_fulltext"] = True
            if data.get("fulltextSource"):
                metadata["fulltext_source"] = data.get("fulltextSource")
        elif data.get("fulltext_attempted"):
            # Extraction was attempted but failed (timeout, empty, etc.)
            # Mark so we don't retry on every incremental update
            metadata["has_fulltext"] = "failed"

        # Add tags as a single string
        if tags := data.get("tags"):
            metadata["tags"] = " ".join([tag.get("tag", "") for tag in tags])
        else:
            metadata["tags"] = ""

        # Add citation key if available
        extra = data.get("extra", "")
        citation_key = ""
        for line in extra.split("\n"):
            if line.lower().startswith(("citation key:", "citationkey:")):
                citation_key = line.split(":", 1)[1].strip()
                break
        metadata["citation_key"] = citation_key

        return metadata

    def should_update_database(self) -> bool:
        """Check if the database should be updated based on configuration."""
        if not self.update_config.get("auto_update", False):
            return False

        frequency = self.update_config.get("update_frequency", "manual")

        if frequency == "manual":
            return False
        elif frequency == "startup":
            return True
        elif frequency == "daily":
            last_update = self.update_config.get("last_update")
            if not last_update:
                return True

            last_update_date = datetime.fromisoformat(last_update)
            return datetime.now() - last_update_date >= timedelta(days=1)
        elif frequency.startswith("every_"):
            try:
                days = int(frequency.split("_")[1])
                last_update = self.update_config.get("last_update")
                if not last_update:
                    return True

                last_update_date = datetime.fromisoformat(last_update)
                return datetime.now() - last_update_date >= timedelta(days=days)
            except (ValueError, IndexError):
                return False

        return False

    def _get_items_from_source(self, limit: int | None = None, extract_fulltext: bool = False, chroma_client: ChromaClient | None = None, force_rebuild: bool = False) -> list[dict[str, Any]]:
        """
        Get items from either local database or API.

        When extract_fulltext=True, requires local mode (ZOTERO_LOCAL=true);
        raises RuntimeError if local mode is not enabled.
        Otherwise uses API (faster, metadata-only).

        Args:
            limit: Optional limit on number of items
            extract_fulltext: Whether to extract fulltext content
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists

        Returns:
            List of items in API-compatible format
        """
        if extract_fulltext:
            if not is_local_mode():
                raise RuntimeError(
                    "Fulltext extraction requires local mode but ZOTERO_LOCAL is not enabled. "
                    "Set ZOTERO_LOCAL=true or run 'zotero-mcp setup' to enable local mode."
                )
            return self._get_items_from_local_db(
                limit,
                extract_fulltext=extract_fulltext,
                chroma_client=chroma_client,
                force_rebuild=force_rebuild
            )
        else:
            return self._get_items_from_api(limit)

    def _get_items_from_local_db(self, limit: int | None = None, extract_fulltext: bool = False, chroma_client: ChromaClient | None = None, force_rebuild: bool = False) -> list[dict[str, Any]]:
        """
        Get items from local Zotero database.

        Args:
            limit: Optional limit on number of items
            extract_fulltext: Whether to extract fulltext content
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists

        Returns:
            List of items in API-compatible format
        """
        logger.info("Fetching items from local Zotero database...")

        try:
            # Load per-run config, including extraction limits and db path if provided
            pdf_max_pages = None
            pdf_timeout = 30
            zotero_db_path = self.db_path  # CLI override takes precedence
            # If semantic_search config file exists, prefer its setting
            try:
                if self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as _f:
                        _cfg = json.load(_f)
                        semantic_cfg = _cfg.get('semantic_search', {})
                        extraction_cfg = semantic_cfg.get('extraction', {})
                        pdf_max_pages = extraction_cfg.get('pdf_max_pages')
                        pdf_timeout = extraction_cfg.get('pdf_timeout', 30)
                        # Use config db_path only if no CLI override
                        if not zotero_db_path:
                            zotero_db_path = semantic_cfg.get('zotero_db_path')
            except Exception:
                pass

            with suppress_stdout(), LocalZoteroReader(db_path=zotero_db_path, pdf_max_pages=pdf_max_pages, pdf_timeout=pdf_timeout) as reader:
                # Phase 1: fetch metadata only (fast)
                sys.stderr.write("Scanning local Zotero database for items...\n")
                local_items = reader.get_items_with_text(limit=limit, include_fulltext=False)
                candidate_count = len(local_items)
                sys.stderr.write(f"Found {candidate_count} candidate items.\n")

                # Optional deduplication: if preprint and journalArticle share a DOI/title, keep journalArticle
                # Build index by (normalized DOI or normalized title)
                def norm(s: str | None) -> str | None:
                    if not s:
                        return None
                    return "".join(s.lower().split())

                key_to_best = {}
                for it in local_items:
                    doi_key = ("doi", norm(getattr(it, "doi", None))) if getattr(it, "doi", None) else None
                    title_key = ("title", norm(getattr(it, "title", None))) if getattr(it, "title", None) else None

                    def consider(k):
                        if not k:
                            return
                        cur = key_to_best.get(k)
                        # Prefer journalArticle over preprint; otherwise keep first
                        if cur is None:
                            key_to_best[k] = it
                        else:
                            prefer_types = {"journalArticle": 2, "preprint": 1}
                            cur_score = prefer_types.get(getattr(cur, "item_type", ""), 0)
                            new_score = prefer_types.get(getattr(it, "item_type", ""), 0)
                            if new_score > cur_score:
                                key_to_best[k] = it

                    consider(doi_key)
                    consider(title_key)

                # If a preprint loses against a journal article for same DOI/title, drop it
                filtered_items = []
                for it in local_items:
                    # If there is a journalArticle alternative for same DOI or title, and this is preprint, drop
                    if getattr(it, "item_type", None) == "preprint":
                        k_doi = ("doi", norm(getattr(it, "doi", None))) if getattr(it, "doi", None) else None
                        k_title = ("title", norm(getattr(it, "title", None))) if getattr(it, "title", None) else None
                        drop = False
                        for k in (k_doi, k_title):
                            if not k:
                                continue
                            best = key_to_best.get(k)
                            if best is not None and best is not it and getattr(best, "item_type", None) == "journalArticle":
                                drop = True
                                break
                        if drop:
                            continue
                    filtered_items.append(it)

                local_items = filtered_items
                total_to_extract = len(local_items)
                if total_to_extract != candidate_count:
                    try:
                        sys.stderr.write(f"After filtering/dedup: {total_to_extract} items to process. Extracting content...\n")
                    except Exception:
                        pass
                else:
                    try:
                        sys.stderr.write("Extracting content...\n")
                    except Exception:
                        pass

                # Phase 2: selectively extract fulltext only when requested
                if extract_fulltext:
                    extracted = 0
                    skipped_existing = 0
                    updated_existing = 0
                    items_to_process = []

                    consecutive_timeouts = 0
                    MAX_CONSECUTIVE_TIMEOUTS = 5
                    _extraction_stopped = False  # Set True when circuit breaker trips

                    total_local = len(local_items)
                    _skipped_pdfs = []  # Collect timeout/error names for summary
                    _skipped_failed = []  # Items skipped because extraction previously failed

                    # Show startup note
                    try:
                        sys.stderr.write(
                            "\n  Note: Most papers take 1-3 seconds. Some larger or complex PDFs\n"
                            "  may take up to 30 seconds. Password-protected or corrupted files\n"
                            "  will be skipped automatically. The system moves on to the next\n"
                            "  paper if a file can't be processed in time.\n\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass

                    # Temporarily suppress local_db logger to prevent timeout warnings
                    # from disrupting the progress line — we collect them ourselves
                    _local_db_logger = logging.getLogger("zotero_mcp.local_db")
                    _prev_level = _local_db_logger.level
                    _local_db_logger.setLevel(logging.CRITICAL)

                    for item_idx, it in enumerate(local_items, 1):
                        # Build display string: Author (Year) — Title
                        title = getattr(it, "title", "") or ""
                        creators = getattr(it, "creators", "") or ""
                        date = getattr(it, "date_added", "") or ""
                        first_author = ""
                        if creators:
                            first_author = creators.split(";")[0].split(",")[0].strip()
                            if first_author:
                                first_author += " et al." if ";" in creators else ""
                        year = ""
                        if date and len(date) >= 4:
                            year = date[:4]
                        citation = ""
                        if first_author and year:
                            citation = f"{first_author} ({year}) — "
                        elif first_author:
                            citation = f"{first_author} — "
                        display = f"{citation}{title}"
                        if len(display) > 60:
                            display = display[:57] + "..."

                        # Single-line progress with \r overwrite
                        # MUST fit within terminal width to prevent wrapping
                        try:
                            try:
                                term_width = os.get_terminal_size().columns
                            except (OSError, ValueError):
                                term_width = 80
                            # Build the line and truncate to terminal width - 1
                            # (- 1 to prevent the cursor from wrapping to next line)
                            max_len = term_width - 1
                            status_parts = []
                            if skipped_existing > 0:
                                status_parts.append(f"{skipped_existing} up to date")
                            if extracted > 0:
                                status_parts.append(f"{extracted} extracted")
                            status = f" ({', '.join(status_parts)})" if status_parts else ""
                            prefix = f"  Processing {item_idx}/{total_local}{status} — "
                            # Truncate display to fit remaining space
                            remaining = max_len - len(prefix) - 3  # -3 for "..."
                            if remaining > 0 and display and len(display) > remaining:
                                display = display[:remaining] + "..."
                            line = f"{prefix}{display or 'working...'}"
                            if len(line) > max_len:
                                line = line[:max_len]
                            sys.stderr.write(f"\r{line}{' ' * max(0, max_len - len(line))}")
                            sys.stderr.flush()
                        except Exception:
                            pass

                        should_extract = True

                        # CHECK IF ITEM ALREADY EXISTS (unless force_rebuild or no client)
                        if chroma_client and not force_rebuild:
                            existing_metadata = chroma_client.get_document_metadata(it.key)
                            if existing_metadata:
                                chroma_has_fulltext = existing_metadata.get("has_fulltext", False)
                                local_has_fulltext = len(reader.get_fulltext_meta_for_item(it.item_id)) > 0

                                # Skip if extraction previously failed AND the item hasn't been
                                # modified since (handles case where user replaces a bad PDF)
                                if chroma_has_fulltext == "failed":
                                    chroma_date = existing_metadata.get("date_modified", "")
                                    item_date = getattr(it, "date_modified", "") or ""
                                    if chroma_date == item_date:
                                        # Same modification date — don't retry failed extraction
                                        should_extract = False
                                        skipped_existing += 1
                                        _skipped_failed.append(display or f"item {it.key}")
                                    else:
                                        # Item was modified since last failure — retry
                                        updated_existing += 1
                                elif not chroma_has_fulltext and local_has_fulltext:
                                    # Document exists but lacks fulltext - we need to update it
                                    updated_existing += 1
                                else:
                                    should_extract = False
                                    skipped_existing += 1

                        if should_extract:
                            # Extract fulltext if item doesn't have it yet
                            # (skip if circuit breaker has tripped)
                            if not getattr(it, "fulltext", None) and not _extraction_stopped:
                                text = reader.extract_fulltext_for_item(it.item_id)
                                # Circuit breaker: stop PDF extraction after consecutive timeouts
                                if isinstance(text, tuple) and len(text) == 2 and text[1] == "timeout":
                                    _skipped_pdfs.append(display or f"item {it.key}")
                                    consecutive_timeouts += 1
                                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                                        logger.warning(
                                            f"Stopping PDF extraction after {MAX_CONSECUTIVE_TIMEOUTS} "
                                            f"consecutive timeouts — remaining items will use metadata only"
                                        )
                                        try:
                                            sys.stderr.write(
                                                f"\n  Warning: PDF extraction stopped after {MAX_CONSECUTIVE_TIMEOUTS} "
                                                f"consecutive timeouts. Remaining items will be indexed with "
                                                f"metadata only (titles, abstracts, authors).\n\n"
                                            )
                                        except Exception:
                                            pass
                                        _extraction_stopped = True
                                    # Don't skip the item — still add it with metadata only
                                    it._fulltext_attempted = True  # Mark so metadata knows extraction was tried
                                else:
                                    # Reset counter on successful extraction
                                    if text:
                                        consecutive_timeouts = 0
                                    if text:
                                        # Support new (text, source) return format
                                        if isinstance(text, tuple) and len(text) == 2:
                                            it.fulltext, it.fulltext_source = text[0], text[1]
                                        else:
                                            it.fulltext = text
                                    else:
                                        # Extraction returned empty — mark as attempted
                                        it._fulltext_attempted = True
                            extracted += 1
                            items_to_process.append(it)

                            # (progress shown inline above via \r)

                    # Restore local_db logger
                    _local_db_logger.setLevel(_prev_level)

                    # Clear progress line and show extraction summary
                    try:
                        sys.stderr.write(f"\r{' ' * 120}\r")  # Clear progress line
                        parts = [f"  Extraction complete: {extracted} items to index"]
                        if skipped_existing > 0:
                            parts.append(f"{skipped_existing} already up to date")
                        sys.stderr.write(", ".join(parts) + "\n")
                        if updated_existing > 0:
                            sys.stderr.write(f"  ({updated_existing} items updated with new fulltext)\n")
                        if _skipped_pdfs:
                            sys.stderr.write(f"  Skipped {len(_skipped_pdfs)} PDF(s) (timed out):\n")
                            for name in _skipped_pdfs:
                                sys.stderr.write(f"    - {name}\n")
                        if _skipped_failed:
                            sys.stderr.write(f"  {len(_skipped_failed)} item(s) skipped (PDF extraction previously failed):\n")
                            for name in _skipped_failed[:5]:  # Show first 5
                                sys.stderr.write(f"    - {name}\n")
                            if len(_skipped_failed) > 5:
                                sys.stderr.write(f"    ... and {len(_skipped_failed) - 5} more\n")
                            sys.stderr.write(f"  (To retry these, run with --force-rebuild)\n")
                    except Exception:
                        pass

                    # Replace local_items with filtered list
                    local_items = items_to_process
                else:
                    # Skip fulltext extraction for faster processing
                    for it in local_items:
                        it.fulltext = None
                        it.fulltext_source = None

                # Convert to API-compatible format
                api_items = []
                for item in local_items:
                    # Create API-compatible item structure
                    api_item = {
                        "key": item.key,
                        "version": 0,  # Local items don't have versions
                        "data": {
                            "key": item.key,
                            "itemType": getattr(item, 'item_type', None) or "journalArticle",
                            "title": item.title or "",
                            "abstractNote": item.abstract or "",
                            "extra": item.extra or "",
                            # Include fulltext only when extracted
                            "fulltext": getattr(item, 'fulltext', None) or "" if extract_fulltext else "",
                            "fulltextSource": getattr(item, 'fulltext_source', None) or "" if extract_fulltext else "",
                            # Flag if extraction was attempted but failed (timeout, empty)
                            "fulltext_attempted": getattr(item, '_fulltext_attempted', False),
                            "dateAdded": item.date_added,
                            "dateModified": item.date_modified,
                            "creators": self._parse_creators_string(item.creators) if item.creators else []
                        }
                    }

                    # Add notes if available
                    if item.notes:
                        api_item["data"]["notes"] = item.notes

                    api_items.append(api_item)

                logger.info(f"Retrieved {len(api_items)} items from local database")
                return api_items

        except Exception as e:
            logger.error(f"Error reading from local database: {e}")
            logger.info("Falling back to API...")
            return self._get_items_from_api(limit)

    def _parse_creators_string(self, creators_str: str) -> list[dict[str, str]]:
        """
        Parse creators string from local DB into API format.

        Args:
            creators_str: String like "Smith, John; Doe, Jane"

        Returns:
            List of creator objects
        """
        if not creators_str:
            return []

        creators = []
        for creator in creators_str.split(';'):
            creator = creator.strip()
            if not creator:
                continue

            if ',' in creator:
                last, first = creator.split(',', 1)
                creators.append({
                    "creatorType": "author",
                    "firstName": first.strip(),
                    "lastName": last.strip()
                })
            else:
                creators.append({
                    "creatorType": "author",
                    "name": creator
                })

        return creators

    def _get_items_from_api(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get items from Zotero API (original implementation).

        Args:
            limit: Optional limit on number of items

        Returns:
            List of items from API
        """
        logger.info("Fetching items from Zotero API...")

        # Fetch items in batches to handle large libraries
        batch_size = 100
        start = 0
        all_items = []

        while True:
            batch_params = {"start": start, "limit": batch_size}
            if limit and len(all_items) >= limit:
                break

            try:
                items = self.zotero_client.items(**batch_params)
            except Exception as e:
                if "Connection refused" in str(e):
                    error_msg = (
                        "Cannot connect to Zotero local API. Please ensure:\n"
                        "1. Zotero is running\n"
                        "2. Local API is enabled in Zotero Preferences > Advanced > Enable HTTP server\n"
                        "3. The local API port (default 23119) is not blocked"
                    )
                    raise Exception(error_msg) from e
                else:
                    raise Exception(f"Zotero API connection error: {e}") from e
            if not items:
                break

            # Filter out attachments and notes by default
            filtered_items = [
                item for item in items
                if item.get("data", {}).get("itemType") not in ["attachment", "note"]
            ]

            all_items.extend(filtered_items)
            start += batch_size

            if len(items) < batch_size:
                break

        if limit:
            all_items = all_items[:limit]

        logger.info(f"Retrieved {len(all_items)} items from API")
        return all_items

    def update_database(self,
                       force_full_rebuild: bool = False,
                       limit: int | None = None,
                       extract_fulltext: bool = False) -> dict[str, Any]:
        """
        Update the semantic search database with Zotero items.

        Args:
            force_full_rebuild: Whether to rebuild the entire database
            limit: Limit number of items to process (for testing)
            extract_fulltext: Whether to extract fulltext content from local database

        Returns:
            Update statistics
        """
        logger.info("Starting database update...")
        start_time = datetime.now()

        stats = {
            "total_items": 0,
            "processed_items": 0,
            "added_items": 0,
            "updated_items": 0,
            "recovered_items": 0,
            "skipped_items": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "duration": None
        }

        try:
            # Reset collection if force rebuild
            if force_full_rebuild:
                logger.info("Force rebuilding database...")
                self.chroma_client.reset_collection()

            # Chunking: safeguard + persistence.
            # The safeguard check only fires on incremental updates — a force
            # rebuild has just wiped the collection, so there are no persisted
            # zmcp_chunking_* keys left to compare against.
            chunking_cfg = self._chunking_config or {}
            if not force_full_rebuild:
                stored_md = self.chroma_client.get_collection_metadata()
                # Raises SystemExit with an actionable pointer to
                # --force-rebuild when the persisted mode / chunk sizes
                # differ from the requested config.
                _check_chunking_compatibility(stored_md, chunking_cfg)

            # Resolve (and validate) chunk params once, up front, so bad
            # config aborts before we touch items. Refreshes the cached
            # values the batch loop reads later.
            if chunking_cfg.get("mode") == "recursive":
                cs, ov = _resolve_chunking_params(
                    chunking_cfg, self.chroma_client.embedding_function
                )
                # Write the resolved (possibly clamped) values back so the
                # per-batch call inside `_process_item_batch` and the
                # collection metadata agree.
                self._chunking_config = {
                    **chunking_cfg,
                    "chunk_size_tokens": cs,
                    "chunk_overlap_tokens": ov,
                }
                persist_md = {
                    "zmcp_chunking_mode": "recursive",
                    "zmcp_chunking_chunk_size_tokens": cs,
                    "zmcp_chunking_chunk_overlap_tokens": ov,
                }
            else:
                persist_md = {"zmcp_chunking_mode": "none"}
            # Write every run so the persisted state is self-healing if it
            # was ever lost; `collection.modify` is O(1) on metadata.
            try:
                self.chroma_client.update_collection_metadata(persist_md)
            except Exception as e:
                # Non-fatal: the safeguard still works on whatever metadata
                # is there next run, and first-ever indexing proceeds.
                logger.warning(
                    f"Could not persist chunking metadata on collection: {e}"
                )

            # Get all items from either local DB or API
            all_items = self._get_items_from_source(
                limit=limit,
                extract_fulltext=extract_fulltext,
                chroma_client=self.chroma_client if not force_full_rebuild else None,
                force_rebuild=force_full_rebuild
            )

            stats["total_items"] = len(all_items)
            logger.info(f"Found {stats['total_items']} items to process")
            # User-friendly progress reporting
            total = stats['total_items'] = len(all_items)
            try:
                sys.stderr.write(f"\nIndexing {total} items...\n\n")
                sys.stderr.flush()
            except Exception:
                pass

            # Process items in batches.
            # Heavy local embedders (qwen3-embedding:4b, bge-m3) buckle at
            # batch=25 because each request is ~500K tokens of matmul;
            # batch=4 keeps per-request work proportionate to 4B-param
            # models on consumer hardware. Small embedders
            # (nomic-embed-text, OpenAI) don't mind the smaller batch.
            # TODO (issue): make provider-aware / configurable.
            batch_size = 4
            seen_items = 0
            _failed_docs = []  # Collect failures for end-of-run retry
            for i in range(0, len(all_items), batch_size):
                batch = all_items[i:i + batch_size]

                # Show per-item progress within this batch
                for item in batch:
                    seen_items += 1
                    title = item.get("data", {}).get("title", "")
                    if title and len(title) > 60:
                        title = title[:57] + "..."
                    pct = int(seen_items / total * 100) if total else 0
                    try:
                        sys.stderr.write(f"\r  [{pct:3d}%] {seen_items}/{total} — {title or 'processing...'}")
                        sys.stderr.flush()
                    except Exception:
                        pass

                batch_stats = self._process_item_batch(batch, force_full_rebuild, _failed_docs)

                stats["processed_items"] += batch_stats["processed"]
                stats["added_items"] += batch_stats["added"]
                stats["updated_items"] += batch_stats["updated"]
                stats["skipped_items"] += batch_stats["skipped"]
                stats["errors"] += batch_stats["errors"]

                logger.info(f"Processed {seen_items}/{total} items (added: {stats['added_items']}, skipped: {stats['skipped_items']})")

            # Retry any documents that failed during the main run
            if _failed_docs:
                try:
                    sys.stderr.write(f"\r{' ' * 120}\r")
                    sys.stderr.write(f"\n  Retrying {len(_failed_docs)} failed items...\n")
                except Exception:
                    pass

                import time as _retry_time
                _retry_time.sleep(1)  # Brief pause before retry

                retry_ok = 0
                retry_fail = 0
                for doc, meta, doc_id in _failed_docs:
                    try:
                        self.chroma_client.upsert_documents([doc], [meta], [doc_id])
                        retry_ok += 1
                        stats["errors"] -= 1  # Remove from error count
                        # Don't classify as added vs updated — when the
                        # original batch failed, the add/update lookup never
                        # ran, so we don't know which category it belongs in.
                        # Track recovered items in their own bucket.
                        stats["recovered_items"] += 1
                    except Exception as e2:
                        retry_fail += 1
                        logger.error(f"Retry failed for {doc_id}: {e2}")

                try:
                    sys.stderr.write(f"  Retry: {retry_ok} recovered, {retry_fail} still failed\n")
                except Exception:
                    pass

            # Clear the progress line and show summary
            try:
                sys.stderr.write(f"\r{' ' * 120}\r")  # Clear line
                summary = (
                    f"  Done: {stats['processed_items']} indexed, "
                    f"{stats['skipped_items']} skipped, "
                    f"{stats['errors']} errors"
                )
                if stats["recovered_items"]:
                    summary += f", {stats['recovered_items']} recovered"
                sys.stderr.write(summary + "\n")
            except Exception:
                pass

            # Update last update time
            self.update_config["last_update"] = datetime.now().isoformat()
            self._save_update_config()

            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            stats["end_time"] = end_time.isoformat()

            logger.info(f"Database update completed in {stats['duration']}")
            return stats

        except Exception as e:
            logger.error(f"Error updating database: {e}")
            stats["error"] = str(e)
            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            return stats

    def _process_item_batch(
        self,
        items: list[dict[str, Any]],
        force_rebuild: bool = False,
        _failed_docs: list | None = None,
    ) -> dict[str, int]:
        """Process a batch of items.

        _failed_docs: optional list (passed by reference from update_database)
        that collects (doc_text, metadata, doc_id) tuples for batches that fail
        mid-run. Without this, the retry path at update_database:839-865 is
        dead code — a NameError raised here would crash the whole reindex,
        making every transient ChromaDB error fatal instead of recoverable.
        """
        stats = {"processed": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0}

        documents = []
        metadatas = []
        ids = []

        chunking_cfg = self._chunking_config or {}
        mode = chunking_cfg.get("mode", "none")
        # Resolve chunk sizes once per batch so errors surface early and the
        # log warning for over-large chunk_size fires at most once per update.
        if mode == "recursive":
            chunk_size, chunk_overlap = _resolve_chunking_params(
                chunking_cfg, self.chroma_client.embedding_function
            )
        else:
            chunk_size = chunk_overlap = 0

        for item in items:
            try:
                item_key = item.get("key", "")
                if not item_key:
                    stats["skipped"] += 1
                    continue

                # Create document text and metadata
                # Always include structured fields; append fulltext when available
                fulltext = item.get("data", {}).get("fulltext", "")
                if fulltext and self._extraction_config.get("strip_boilerplate", True):
                    from .text_filters import strip_boilerplate
                    fulltext = strip_boilerplate(
                        fulltext,
                        skip_to_abstract=self._extraction_config.get("skip_to_abstract", True),
                    )
                structured_text = self._create_document_text(item)
                if fulltext.strip():
                    doc_text = (structured_text + "\n\n" + fulltext) if structured_text.strip() else fulltext
                else:
                    doc_text = structured_text
                metadata = self._create_metadata(item)

                if not doc_text.strip():
                    stats["skipped"] += 1
                    continue

                if mode == "none":
                    # Byte-identical to pre-#3 behaviour: truncate to the
                    # embedder's token limit before write.
                    doc_text = self.chroma_client.truncate_text(doc_text)
                    documents.append(doc_text)
                    metadatas.append(metadata)
                    ids.append(item_key)
                    stats["processed"] += 1
                else:
                    # Chunked indexing: split `doc_text` and emit one row per
                    # chunk. The full text is NOT truncated here — each chunk
                    # is sized (<= 60% of the embedder's context window) so
                    # the embedder can consume it without server-side
                    # truncation. Empty output falls back to a single row so
                    # metadata-only items (no abstract, failed extraction)
                    # still land in the index.
                    from .chunker import Chunk, recursive_split
                    chunks = recursive_split(
                        doc_text,
                        chunk_size_tokens=chunk_size,
                        overlap_tokens=chunk_overlap,
                    ) or [Chunk(doc_text, 0)]
                    total = len(chunks)
                    for c in chunks:
                        chunk_meta = dict(metadata)
                        chunk_meta["parent_item_key"] = item_key
                        chunk_meta["chunk_index"] = c.index
                        chunk_meta["chunk_total"] = total
                        documents.append(c.text)
                        metadatas.append(chunk_meta)
                        ids.append(f"{item_key}:c{c.index:03d}")
                    stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1

        # Add documents to ChromaDB if any
        if documents:
            existing_ids = set()
            if not force_rebuild:
                existing_ids = self.chroma_client.get_existing_ids(ids)

            try:
                self.chroma_client.upsert_documents(documents, metadatas, ids)
                if mode == "recursive":
                    # Count added/updated per parent item, not per chunk, so
                    # the user-facing stats (`added_items`, `updated_items`)
                    # still reflect Zotero items — consistent with the
                    # mode='none' path and with what the CLI prints.
                    per_parent: dict[str, list[str]] = {}
                    for doc_id, meta in zip(ids, metadatas):
                        parent = meta.get("parent_item_key") or doc_id
                        per_parent.setdefault(parent, []).append(doc_id)
                    for chunk_ids in per_parent.values():
                        if all(cid in existing_ids for cid in chunk_ids):
                            stats["updated"] += 1
                        else:
                            stats["added"] += 1
                else:
                    for doc_id in ids:
                        if doc_id in existing_ids:
                            stats["updated"] += 1
                        else:
                            stats["added"] += 1
            except Exception as e:
                # Batch failed — collect failures for end-of-run retry.
                # ChromaDB's ONNX tokenizer can fail intermittently in bursts;
                # retrying immediately usually fails too. Collecting failures
                # and retrying after all batches are done is more effective.
                logger.warning(f"Batch upsert failed ({e}), saving for retry")
                if _failed_docs is not None:
                    for j in range(len(documents)):
                        _failed_docs.append((documents[j], metadatas[j], ids[j]))
                    # Count them as errors so stats are accurate
                    stats["errors"] += len(documents)
                else:
                    # No retry list — this is the legacy crash path; re-raise
                    # so caller sees the real error instead of hiding it.
                    raise

        return stats

    def search(self,
               query: str,
               limit: int = 10,
               filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Perform semantic search over the Zotero library.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            Search results with Zotero item details
        """
        try:
            # Chunking mode drives both over-fetch and the aggregator path.
            # When off, the search path is byte-identical to pre-#3.
            chunking_mode = (self._chunking_config or {}).get("mode", "none")
            chunking_active = chunking_mode == "recursive"

            # Over-fetch candidates when re-ranking is enabled
            reranker = self._get_reranker()
            fetch_limit = limit
            multiplier = 1
            if reranker:
                multiplier = self._reranker_config.get("candidate_multiplier", 3)
                fetch_limit = limit * multiplier
            if chunking_active:
                # Spec: fetch more candidates so the aggregator has enough
                # chunks to produce `limit` unique parents. Capped at 100
                # to keep single-query cost bounded.
                fetch_limit = max(fetch_limit, min(limit * 10, 100))

            # Perform semantic search
            results = self.chroma_client.search(
                query_texts=[query],
                n_results=fetch_limit,
                where=filters
            )

            if chunking_active:
                # Aggregate chunk hits to one per parent paper, then
                # optionally rerank, then enrich via parent_item_key.
                hits = _chroma_results_to_hits(results)
                agg_target = limit * multiplier if reranker else limit
                hits = _aggregate_chunk_hits(hits, agg_target)
                if reranker and hits:
                    documents = [h.get("document", "") for h in hits]
                    ranked_indices = reranker.rerank(query, documents, top_k=limit)
                    hits = [hits[i] for i in ranked_indices]
                else:
                    hits = hits[:limit]
                enriched_results = self._enrich_hits(hits, query)
            else:
                # Re-rank results with cross-encoder if enabled
                if reranker and results.get("documents") and results["documents"][0]:
                    documents = results["documents"][0]
                    ranked_indices = reranker.rerank(query, documents, top_k=limit)
                    for key in ["ids", "distances", "documents", "metadatas"]:
                        if results.get(key) and results[key][0]:
                            results[key][0] = [results[key][0][i] for i in ranked_indices]

                # Enrich results with full Zotero item data
                enriched_results = self._enrich_search_results(results, query)

            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": enriched_results,
                "total_found": len(enriched_results)
            }

        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": [],
                "total_found": 0,
                "error": str(e)
            }

    def _enrich_search_results(self, chroma_results: dict[str, Any], query: str) -> list[dict[str, Any]]:
        """Enrich ChromaDB results with full Zotero item data."""
        enriched = []

        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return enriched

        ids = chroma_results["ids"][0]
        distances = chroma_results.get("distances", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]

        for i, item_key in enumerate(ids):
            try:
                # Get full item data from Zotero
                zotero_item = self.zotero_client.item(item_key)

                enriched_result = {
                    "item_key": item_key,
                    "similarity_score": 1 - distances[i] if i < len(distances) else 0,
                    "matched_text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "zotero_item": zotero_item,
                    "query": query
                }

                enriched.append(enriched_result)

            except Exception as e:
                logger.error(f"Error enriching result for item {item_key}: {e}")
                # Include basic result even if enrichment fails
                enriched.append({
                    "item_key": item_key,
                    "similarity_score": 1 - distances[i] if i < len(distances) else 0,
                    "matched_text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "query": query,
                    "error": f"Could not fetch full item data: {e}"
                })

        return enriched

    def _enrich_hits(self, hits: list[dict], query: str) -> list[dict[str, Any]]:
        """Enrich aggregated chunk hits with full Zotero item data.

        Parallel to `_enrich_search_results` but works from per-hit dicts
        (already aggregated by `_aggregate_chunk_hits`) and uses
        `parent_item_key` — not the chunk-level id — to look up the paper
        in Zotero. Adds `matched_chunk_text` and `matched_chunk_index`
        surfacing the best-scoring passage per paper.
        """
        enriched: list[dict[str, Any]] = []
        for h in hits:
            meta = h.get("metadata") or {}
            parent_key = meta.get("parent_item_key")
            if not parent_key:
                # Row pre-dates chunking — fall back to the chunk id.
                doc_id = h.get("id", "") or ""
                parent_key = doc_id.split(":c")[0] if ":c" in doc_id else doc_id
            dist = h.get("distance")
            similarity = (1 - dist) if isinstance(dist, (int, float)) else 0
            base = {
                "item_key": parent_key,
                "similarity_score": similarity,
                "matched_text": h.get("document", ""),
                "matched_chunk_text": h.get("matched_chunk_text", h.get("document", "")),
                "matched_chunk_index": h.get("matched_chunk_index", meta.get("chunk_index", 0)),
                "metadata": meta,
                "query": query,
            }
            try:
                zotero_item = self.zotero_client.item(parent_key)
                base["zotero_item"] = zotero_item
            except Exception as e:
                logger.error(f"Error enriching result for item {parent_key}: {e}")
                base["error"] = f"Could not fetch full item data: {e}"
            enriched.append(base)
        return enriched

    def get_database_status(self) -> dict[str, Any]:
        """Get status information about the semantic search database."""
        collection_info = self.chroma_client.get_collection_info()

        return {
            "collection_info": collection_info,
            "update_config": self.update_config,
            "should_update": self.should_update_database(),
            "last_update": self.update_config.get("last_update"),
        }

    def delete_item(self, item_key: str) -> bool:
        """Delete an item from the semantic search database."""
        try:
            self.chroma_client.delete_documents([item_key])
            return True
        except Exception as e:
            logger.error(f"Error deleting item {item_key}: {e}")
            return False


def create_semantic_search(config_path: str | None = None, db_path: str | None = None) -> ZoteroSemanticSearch:
    """
    Create a ZoteroSemanticSearch instance.

    Args:
        config_path: Path to configuration file
        db_path: Optional path to Zotero database (overrides config file)

    Returns:
        Configured ZoteroSemanticSearch instance
    """
    return ZoteroSemanticSearch(config_path=config_path, db_path=db_path)
