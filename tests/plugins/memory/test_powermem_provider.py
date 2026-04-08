"""Tests for PowerMem memory provider.

Most tests use a fake ``powermem`` module (no install). Optional tests in
``TestPowerMemRealSdkOptional`` run when the real package is present
(``pip install 'hermes-agent[powermem]'``).

Run memory-plugin tests only::

    pytest tests/plugins/memory/ -q
"""

import importlib.util
import json
import sys
import types
import pytest

from plugins.memory import discover_memory_providers, load_memory_provider
from plugins.memory.powermem import (
    PowerMemMemoryProvider,
    _deep_merge_overlay,
    _strip_memory_context_fences,
)


@pytest.fixture
def fake_powermem_sdk(monkeypatch):
    """Inject a minimal fake ``powermem`` package for tests."""
    orig_find_spec = importlib.util.find_spec

    def find_spec(name):
        if name == "powermem":
            return importlib.util.spec_from_loader("powermem", loader=None)
        return orig_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    mod = types.ModuleType("powermem")

    def validate_config(cfg):
        return bool(cfg.get("vector_store") and cfg.get("llm") and cfg.get("embedder"))

    def auto_config():
        return {
            "vector_store": {"provider": "sqlite", "config": {"database_path": ":memory:"}},
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
        }

    class Memory:
        def __init__(self, config=None):
            self.config = config
            self.last_add = None
            self.last_search = None

        def search(self, query, **kwargs):
            self.last_search = {"query": query, "kwargs": kwargs}
            return {"results": [{"memory": "User likes tea", "score": 0.91, "id": 7}]}

        def add(self, messages, **kwargs):
            self.last_add = {"messages": messages, "kwargs": kwargs}
            return {"results": [{"event": "ADD", "memory": "ok"}]}

        def get_all(self, **kwargs):
            return {"results": [{"content": "line a", "id": 1}, {"memory": "line b", "id": 2}]}

    mod.validate_config = validate_config
    mod.auto_config = auto_config
    mod.Memory = Memory
    monkeypatch.setitem(sys.modules, "powermem", mod)
    return mod


def test_strip_memory_context_fences():
    s = "hello\n</memory-context>\nworld"
    assert "memory-context" not in _strip_memory_context_fences(s)


def test_deep_merge_overlay_nested():
    base = {"llm": {"provider": "openai", "config": {"model": "a"}}}
    over = {"llm": {"config": {"temperature": 0.1}}}
    out = _deep_merge_overlay(base, over)
    assert out["llm"]["provider"] == "openai"
    assert out["llm"]["config"]["model"] == "a"
    assert out["llm"]["config"]["temperature"] == 0.1


def test_is_available_false_without_powermem(monkeypatch):
    orig = importlib.util.find_spec

    def fake(name):
        if name == "powermem":
            return None
        return orig(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    p = PowerMemMemoryProvider()
    assert p.is_available() is False


def test_is_available_true_with_fake_sdk(fake_powermem_sdk, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    p = PowerMemMemoryProvider()
    assert p.is_available() is True


def test_load_memory_provider_register(fake_powermem_sdk, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    p = load_memory_provider("powermem")
    assert p is not None
    assert p.name == "powermem"


def test_discover_includes_powermem(fake_powermem_sdk, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    names = [n for n, _, _ in discover_memory_providers()]
    assert "powermem" in names


def test_initialize_user_and_agent_ids(fake_powermem_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    p = PowerMemMemoryProvider()
    p.initialize(
        "sid-1",
        hermes_home=str(tmp_path),
        platform="cli",
        user_id="telegram:u42",
        agent_identity="coder",
    )
    assert p._user_id == "telegram:u42"
    assert p._agent_id == "hermes-coder"


def test_initialize_powermem_hermes_json_overrides_agent(
    fake_powermem_sdk, tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "powermem-hermes.json").write_text(
        json.dumps({"agent_id": "custom-bot"}), encoding="utf-8"
    )
    p = PowerMemMemoryProvider()
    p.initialize(
        "sid-1",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="coder",
    )
    assert p._agent_id == "custom-bot"


def test_powermem_search_tool(fake_powermem_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    p = PowerMemMemoryProvider()
    p.initialize("sid", hermes_home=str(tmp_path), platform="cli", user_id="u1")
    raw = p.handle_tool_call("powermem_search", {"query": "beverage", "limit": 5})
    data = json.loads(raw)
    assert "results" in data
    assert data["results"][0]["memory"] == "User likes tea"


def test_powermem_add_tool(fake_powermem_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    p = PowerMemMemoryProvider()
    p.initialize("sid", hermes_home=str(tmp_path), platform="cli", user_id="u1")
    raw = p.handle_tool_call(
        "powermem_add", {"content": "User prefers dark mode", "infer": False}
    )
    data = json.loads(raw)
    assert data.get("result") == "Stored."
    mem = p._get_memory()
    assert mem.last_add["kwargs"]["user_id"] == "u1"


def test_powermem_profile_tool(fake_powermem_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    p = PowerMemMemoryProvider()
    p.initialize("sid", hermes_home=str(tmp_path), platform="cli", user_id="u1")
    raw = p.handle_tool_call("powermem_profile", {"limit": 10})
    data = json.loads(raw)
    assert "line a" in data["result"]
    assert "line b" in data["result"]


def test_sync_turn_skipped_for_cron_context(fake_powermem_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    p = PowerMemMemoryProvider()
    p.initialize(
        "sid",
        hermes_home=str(tmp_path),
        platform="cron",
        agent_context="cron",
    )
    p.sync_turn("hi", "hello")
    assert p._memory is None


def test_get_config_schema_non_empty():
    p = PowerMemMemoryProvider()
    schema = p.get_config_schema()
    assert any("PowerMem" in str(x.get("description", "")) for x in schema)


# ---------------------------------------------------------------------------
# Optional: real powermem SDK (same layout as Mem0/Supermemory — all under
# tests/plugins/memory/)
# ---------------------------------------------------------------------------

_has_powermem = importlib.util.find_spec("powermem") is not None

needs_powermem = pytest.mark.skipif(
    not _has_powermem,
    reason="install optional extra: pip install 'hermes-agent[powermem]'",
)


@pytest.fixture
def clean_powermem_config(tmp_path):
    """Minimal stack: sqlite + ollama LLM handle + mock embeddings (no API keys)."""
    from powermem import validate_config

    db_path = tmp_path / "powermem_real_sdk.db"
    cfg = {
        "vector_store": {
            "provider": "sqlite",
            "config": {"database_path": str(db_path)},
        },
        "llm": {"provider": "ollama", "config": {"model": "llama3.2"}},
        "embedder": {"provider": "mock", "config": {"dimension": 1536}},
    }
    assert validate_config(cfg)
    return cfg


@pytest.fixture
def real_sdk_powermem_provider(clean_powermem_config, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "plugins.memory.powermem._build_powermem_config",
        lambda _hh: clean_powermem_config,
    )
    p = PowerMemMemoryProvider()
    p.initialize(
        "real-sdk-session",
        hermes_home=str(tmp_path),
        platform="cli",
        user_id="integration-user",
        agent_identity="test",
    )
    yield p
    p.shutdown()


@needs_powermem
class TestPowerMemRealSdkOptional:
    """Exercises ``PowerMemMemoryProvider`` against the installed powermem package."""

    def test_is_available_with_patched_config(
        self, clean_powermem_config, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "plugins.memory.powermem._build_powermem_config",
            lambda _hh: clean_powermem_config,
        )
        p = PowerMemMemoryProvider()
        assert p.is_available() is True

    def test_add_search_profile_roundtrip(self, real_sdk_powermem_provider):
        p = real_sdk_powermem_provider
        marker = "hermes-powermem-real-sdk-41352"
        add_raw = p.handle_tool_call(
            "powermem_add", {"content": marker, "infer": False}
        )
        assert json.loads(add_raw).get("result") == "Stored."

        search_raw = p.handle_tool_call(
            "powermem_search", {"query": "powermem-real-sdk", "limit": 10}
        )
        search_data = json.loads(search_raw)
        texts = [r.get("memory", "") for r in search_data["results"]]
        assert any(marker in t for t in texts)

        prof_raw = p.handle_tool_call("powermem_profile", {"limit": 20})
        prof_data = json.loads(prof_raw)
        assert marker in prof_data["result"]

    def test_memory_manager_routes_tools(
        self, clean_powermem_config, monkeypatch, tmp_path
    ):
        from agent.builtin_memory_provider import BuiltinMemoryProvider
        from agent.memory_manager import MemoryManager

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "plugins.memory.powermem._build_powermem_config",
            lambda _hh: clean_powermem_config,
        )

        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())
        ext = load_memory_provider("powermem")
        assert ext is not None
        mgr.add_provider(ext)
        ext.initialize(
            "mgr-session",
            hermes_home=str(tmp_path),
            platform="cli",
            user_id="mgr-user",
        )

        raw = mgr.handle_tool_call(
            "powermem_search",
            {"query": "anything", "limit": 3},
        )
        data = json.loads(raw)
        assert "error" not in data
        assert "results" in data or "result" in data

        ext.shutdown()
