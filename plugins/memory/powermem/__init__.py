"""PowerMem memory plugin — MemoryProvider interface.

Uses the PowerMem Python SDK (https://github.com/oceanbase/powermem) for
persistent agent memory: hybrid retrieval, optional graph store, and
intelligent extraction on add.

Configuration:
  - Recommended: copy PowerMem `.env` keys into ``$HERMES_HOME/.env``, or
    place a full JSON config at ``$HERMES_HOME/powermem.json`` (merged over
    env-derived defaults).
  - ``validate_config`` from PowerMem requires ``vector_store``, ``llm``, and
    ``embedder`` sections — see PowerMem docs and ``pmem config init``.

Hermes activates this provider when ``memory.provider: powermem`` in config.yaml.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

_MEMORY_CONTEXT_FENCE_RE = re.compile(
    r"</?\s*memory-context\s*>", re.IGNORECASE
)


def _strip_memory_context_fences(text: str) -> str:
    """Remove memory-context fence markers from strings we send to PowerMem."""
    if not text:
        return text
    return _MEMORY_CONTEXT_FENCE_RE.sub("", text).strip()


def _deep_merge_overlay(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Merge powermem.json over auto_config (recursive for nested dicts)."""
    out = dict(base)
    for key, val in (overlay or {}).items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge_overlay(out[key], val)
        else:
            out[key] = val
    return out


def _load_powermem_json(hermes_home: str) -> Dict[str, Any]:
    path = Path(hermes_home) / "powermem.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read powermem.json: %s", e)
        return {}


def _build_powermem_config(hermes_home: str) -> Dict[str, Any]:
    """Load dotenv from HERMES_HOME, auto_config from env, then merge powermem.json."""
    from dotenv import load_dotenv

    load_dotenv(Path(hermes_home) / ".env", override=False)
    from powermem import auto_config

    cfg = auto_config()
    overlay = _load_powermem_json(hermes_home)
    if overlay:
        cfg = _deep_merge_overlay(cfg, overlay)
    return cfg


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "powermem_search",
    "description": (
        "Search PowerMem long-term memory by meaning and keywords. "
        "Returns ranked memory snippets with relevance scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {
                "type": "integer",
                "description": "Max results (default 10, max 50).",
            },
        },
        "required": ["query"],
    },
}

ADD_SCHEMA = {
    "name": "powermem_add",
    "description": (
        "Store content in PowerMem long-term memory. "
        "With infer=true (default), PowerMem may extract and consolidate facts; "
        "with infer=false, stores the text directly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Text to remember."},
            "infer": {
                "type": "boolean",
                "description": "Use intelligent extraction (default true).",
            },
        },
        "required": ["content"],
    },
}

PROFILE_SCHEMA = {
    "name": "powermem_profile",
    "description": (
        "List stored PowerMem memories for this user (up to a limit). "
        "Use for a broad overview; use powermem_search for targeted recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max memories to return (default 50, max 200).",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider
# ---------------------------------------------------------------------------


class PowerMemMemoryProvider(MemoryProvider):
    """PowerMem-backed long-term memory."""

    def __init__(self):
        self._memory = None  # powermem.Memory
        self._memory_lock = threading.Lock()
        self._hermes_home = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._skip_writes = False
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "powermem"

    def is_available(self) -> bool:
        if importlib.util.find_spec("powermem") is None:
            return False
        try:
            from hermes_constants import get_hermes_home
            from powermem import validate_config

            hh = str(get_hermes_home())
            cfg = _build_powermem_config(hh)
            return bool(validate_config(cfg))
        except Exception:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "setup_note",
                "description": (
                    "PowerMem needs vector_store + llm + embedder in "
                    "$HERMES_HOME/.env (see PowerMem docs) or a full "
                    "$HERMES_HOME/powermem.json. Run: pip install powermem && pmem config init"
                ),
                "required": False,
                "url": "https://github.com/oceanbase/powermem",
            },
            {
                "key": "agent_id",
                "description": "Agent id stored with memories (default hermes)",
                "default": "hermes",
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Persist optional hermes-side defaults in powermem-hermes.json."""
        agent_id = (values or {}).get("agent_id", "").strip()
        if not agent_id:
            return
        path = Path(hermes_home) / "powermem-hermes.json"
        try:
            existing = {}
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
            existing["agent_id"] = agent_id
            path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save powermem-hermes.json: %s", e)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._hermes_home = str(kwargs.get("hermes_home", ""))
        agent_context = kwargs.get("agent_context", "") or ""
        platform = kwargs.get("platform", "cli")
        if agent_context in ("cron", "flush") or platform == "cron":
            self._skip_writes = True
            logger.debug(
                "PowerMem writes disabled: agent_context=%s platform=%s",
                agent_context,
                platform,
            )

        from hermes_constants import get_hermes_home

        if not self._hermes_home:
            self._hermes_home = str(get_hermes_home())

        self._user_id = kwargs.get("user_id") or "hermes-user"
        # Per-profile agent isolation (overridable via powermem-hermes.json)
        self._agent_id = "hermes"
        ident = (kwargs.get("agent_identity") or "").strip()
        if ident:
            self._agent_id = f"hermes-{ident}"
        aid_path = Path(self._hermes_home) / "powermem-hermes.json"
        if aid_path.exists():
            try:
                raw = json.loads(aid_path.read_text(encoding="utf-8"))
                if raw.get("agent_id"):
                    self._agent_id = str(raw["agent_id"])
            except Exception:
                pass

        with self._memory_lock:
            self._memory = None

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "PowerMem circuit breaker tripped after %d failures; pausing %ds",
                self._consecutive_failures,
                _BREAKER_COOLDOWN_SECS,
            )

    def _get_memory(self):
        with self._memory_lock:
            if self._memory is not None:
                return self._memory
            from powermem import Memory

            cfg = _build_powermem_config(self._hermes_home)
            self._memory = Memory(config=cfg)
            return self._memory

    @staticmethod
    def _memory_text(item: Dict[str, Any]) -> str:
        return (item.get("memory") or item.get("content") or "").strip()

    def system_prompt_block(self) -> str:
        return (
            "# PowerMem\n"
            f"Active. Scoped user_id={self._user_id!r}, agent_id={self._agent_id!r}.\n"
            "Use powermem_search for recall, powermem_add to store facts, "
            "powermem_profile for a listing."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## PowerMem\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._skip_writes or self._is_breaker_open():
            return

        def _run():
            try:
                mem = self._get_memory()
                q = _strip_memory_context_fences(query or "")
                if not q:
                    return
                out = mem.search(
                    q,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    limit=5,
                )
                rows = out.get("results") or []
                if rows:
                    lines = []
                    for r in rows:
                        t = self._memory_text(r)
                        if t:
                            lines.append(f"- {t}")
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("PowerMem prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="powermem-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        if self._skip_writes or self._is_breaker_open():
            return

        u = _strip_memory_context_fences(user_content or "")
        a = _strip_memory_context_fences(assistant_content or "")
        if not u and not a:
            return

        def _sync():
            try:
                mem = self._get_memory()
                messages = []
                if u:
                    messages.append({"role": "user", "content": u})
                if a:
                    messages.append({"role": "assistant", "content": a})
                mem.add(
                    messages,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=True,
                )
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("PowerMem sync_turn failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="powermem-sync"
        )
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_SCHEMA, PROFILE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": (
                    "PowerMem temporarily unavailable after repeated errors. "
                    "Will recover automatically."
                )
            })

        if self._skip_writes and tool_name == "powermem_add":
            return tool_error("PowerMem add is disabled in this agent context.")

        try:
            mem = self._get_memory()
        except Exception as e:
            return tool_error(str(e))

        try:
            if tool_name == "powermem_search":
                query = (args.get("query") or "").strip()
                if not query:
                    return tool_error("Missing required parameter: query")
                limit = min(int(args.get("limit", 10)), 50)
                out = mem.search(
                    query,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    limit=limit,
                )
                self._record_success()
                results = []
                for r in out.get("results") or []:
                    t = self._memory_text(r)
                    if not t:
                        continue
                    results.append({
                        "memory": t,
                        "score": r.get("score", 0),
                        "id": r.get("id"),
                    })
                if not results:
                    return json.dumps({"result": "No matching memories found."})
                return json.dumps({"results": results, "count": len(results)})

            if tool_name == "powermem_add":
                content = _strip_memory_context_fences(args.get("content") or "")
                if not content:
                    return tool_error("Missing required parameter: content")
                infer = bool(args.get("infer", True))
                out = mem.add(
                    content,
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    infer=infer,
                )
                self._record_success()
                return json.dumps({"result": "Stored.", "raw": out})

            if tool_name == "powermem_profile":
                limit = min(int(args.get("limit", 50)), 200)
                out = mem.get_all(
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    limit=limit,
                    offset=0,
                )
                self._record_success()
                lines = []
                for r in out.get("results") or []:
                    t = self._memory_text(r)
                    if t:
                        lines.append(t)
                if not lines:
                    return json.dumps({"result": "No memories stored yet.", "count": 0})
                return json.dumps({
                    "result": "\n".join(f"- {x}" for x in lines),
                    "count": len(lines),
                })

            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            self._record_failure()
            return tool_error(f"PowerMem error: {e}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._memory_lock:
            self._memory = None


def register(ctx) -> None:
    """Register PowerMem as a memory provider plugin."""
    ctx.register_memory_provider(PowerMemMemoryProvider())
