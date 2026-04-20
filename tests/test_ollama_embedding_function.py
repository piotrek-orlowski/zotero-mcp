"""Tests for the native Ollama embedding provider (issue #1)."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest
import requests


pytest.importorskip("chromadb", reason="semantic extra not installed")

from zotero_mcp.chroma_client import (  # noqa: E402
    OllamaEmbeddingFunction,
    OLLAMA_DEFAULT_CONTEXT_WINDOW,
)


def _fake_response(status: int, payload: dict | None = None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.json.return_value = payload or {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def _as_python_lists(result):
    """ChromaDB's EmbeddingFunction.__call__ wraps our return value with
    normalize_embeddings, which coerces to numpy arrays. Convert back for
    plain-list comparisons in tests."""
    return [list(row.tolist() if hasattr(row, "tolist") else row)
            for row in result]


def test_call_batched_roundtrip():
    func = OllamaEmbeddingFunction(model_name="nomic-embed-text")
    payload = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    with patch("zotero_mcp.chroma_client.requests.post",
               return_value=_fake_response(200, payload)) as post:
        out = func(["hello", "world"])
    assert _as_python_lists(out) == [[pytest.approx(0.1), pytest.approx(0.2)],
                                     [pytest.approx(0.3), pytest.approx(0.4)]]
    (args, kwargs) = post.call_args
    assert args[0] == "http://localhost:11434/api/embed"
    assert kwargs["json"] == {"model": "nomic-embed-text",
                              "input": ["hello", "world"]}
    assert kwargs["timeout"] == 60


def test_call_empty_input_makes_no_http_call():
    """Empty input must short-circuit without POSTing to Ollama.

    ChromaDB's EmbeddingFunction base class wraps __call__ and rejects
    empty-list returns with ValueError (validate_embeddings). That's fine —
    the invariant we care about is that our code never talks to the Ollama
    server for empty batches, regardless of how the wrapper then reacts.
    """
    func = OllamaEmbeddingFunction(model_name="nomic-embed-text")
    with patch("zotero_mcp.chroma_client.requests.post") as post:
        try:
            func([])
        except ValueError:
            pass  # ChromaDB wrapper rejects empty; unrelated to our logic.
    post.assert_not_called()


def test_call_model_not_pulled_raises():
    func = OllamaEmbeddingFunction(model_name="bge-m3")
    with patch("zotero_mcp.chroma_client.requests.post",
               return_value=_fake_response(404, {})):
        with pytest.raises(RuntimeError, match="ollama pull bge-m3"):
            func(["doc"])


def test_call_unexpected_response_raises():
    func = OllamaEmbeddingFunction(model_name="nomic-embed-text")
    with patch("zotero_mcp.chroma_client.requests.post",
               return_value=_fake_response(200, {"oops": []})):
        with pytest.raises(RuntimeError, match="Unexpected Ollama response"):
            func(["doc"])


def test_truncate_uses_context_window():
    nomic = OllamaEmbeddingFunction(model_name="nomic-embed-text")
    mini = OllamaEmbeddingFunction(model_name="all-minilm")
    unknown = OllamaEmbeddingFunction(model_name="some-random-model")

    long = "x" * 100_000

    assert len(nomic.truncate([long])[0]) == 24576
    assert len(mini.truncate([long])[0]) == 768
    assert len(unknown.truncate([long])[0]) == OLLAMA_DEFAULT_CONTEXT_WINDOW * 3

    # And the property reflects the same numbers.
    assert nomic.embedding_max_tokens == 8192
    assert mini.embedding_max_tokens == 256
    assert unknown.embedding_max_tokens == OLLAMA_DEFAULT_CONTEXT_WINDOW


def test_model_name_with_tag():
    func = OllamaEmbeddingFunction(model_name="nomic-embed-text:latest")
    assert func.embedding_max_tokens == 8192


def test_qwen3_embedding_context_window():
    # Native max_position_embeddings across all sizes is 32768 (per HF cards),
    # and the ":<size>" tag should be stripped before dict lookup.
    for tag in ("qwen3-embedding", "qwen3-embedding:0.6b",
                "qwen3-embedding:4b", "qwen3-embedding:8b"):
        assert OllamaEmbeddingFunction(model_name=tag).embedding_max_tokens == 32768


def test_host_trailing_slash_normalized():
    func = OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        host="http://localhost:11434/",
    )
    assert func.host == "http://localhost:11434"
    with patch("zotero_mcp.chroma_client.requests.post",
               return_value=_fake_response(200, {"embeddings": [[0.0]]})) as post:
        func(["x"])
    (args, _) = post.call_args
    assert args[0] == "http://localhost:11434/api/embed"


# --- Branch routing in _create_embedding_function ------------------------


def _make_client(embedding_config, monkeypatch_env=None):
    """Instantiate ChromaClient purely to exercise _create_embedding_function
    without touching disk or ChromaDB. We only need the embedding function it
    builds, so bypass the __init__ that spins up a PersistentClient."""
    from zotero_mcp.chroma_client import ChromaClient
    obj = ChromaClient.__new__(ChromaClient)
    obj.embedding_model = "ollama"
    obj.embedding_config = embedding_config or {}
    obj._config = {}  # not used by the ollama branch, but safe to set
    return obj._create_embedding_function()


def test_branch_uses_config_over_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "other-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://env-host:9999")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")

    func = _make_client({
        "model_name": "bge-m3",
        "host": "http://cfg-host:11434",
        "timeout": 120,
    })
    assert isinstance(func, OllamaEmbeddingFunction)
    assert func.model_name == "bge-m3"
    assert func.host == "http://cfg-host:11434"
    assert func.timeout == 120


def test_branch_uses_env_over_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("OLLAMA_HOST", "http://env-host:9999")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "42")

    func = _make_client({})
    assert func.model_name == "mxbai-embed-large"
    assert func.host == "http://env-host:9999"
    assert func.timeout == 42


def test_branch_uses_default_when_neither_set(monkeypatch):
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_TIMEOUT", raising=False)

    func = _make_client({})
    assert func.model_name == "nomic-embed-text"
    assert func.host == "http://localhost:11434"
    assert func.timeout == 60


# --- create_chroma_client config wiring ----------------------------------


def test_create_chroma_client_reads_ollama_section(monkeypatch, tmp_path):
    """semantic_search.ollama block in config.json should flow into the
    embedding_config seen by the Ollama branch. Also verifies precedence
    over env vars."""
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "env-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://env-host:9999")

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "semantic_search": {
            "collection_name": "zotero_library_test",
            "embedding_model": "ollama",
            "ollama": {
                "model": "bge-m3",
                "host": "http://cfg-host:11434",
                "timeout": 77,
            },
        }
    }))

    # We only care about the ChromaClient init args, not the actual ChromaDB
    # startup — patch it out.
    captured = {}

    def _fake_init(self, **kwargs):
        captured.update(kwargs)
        # Do not call real __init__; we only want to capture args.
        self.embedding_model = kwargs.get("embedding_model")
        self.embedding_config = kwargs.get("embedding_config", {})

    from zotero_mcp import chroma_client as mod
    with patch.object(mod.ChromaClient, "__init__", _fake_init):
        client = mod.create_chroma_client(config_path=str(cfg_file))

    assert captured["embedding_model"] == "ollama"
    ec = captured["embedding_config"]
    assert ec["model_name"] == "bge-m3"  # config wins over env
    assert ec["host"] == "http://cfg-host:11434"
    assert ec["timeout"] == 77
    assert client.embedding_model == "ollama"
