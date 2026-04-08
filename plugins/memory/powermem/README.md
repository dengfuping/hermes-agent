# PowerMem Memory Provider

Hybrid long-term memory (vector + full-text + optional graph) via the [PowerMem Python SDK](https://github.com/oceanbase/powermem). LLM-assisted extraction on add; storage and backends are configured in PowerMem (e.g. local SQLite, or OceanBase / SeekDB for production).

## Requirements

- `pip install 'hermes-agent[powermem]'` or `pip install powermem`
- A valid PowerMem configuration: `vector_store`, `llm`, and `embedder` must pass upstream `validate_config()` (see PowerMem docs and `pmem config init`).

## Setup

```bash
hermes memory setup    # select "powermem"
```

Or manually:

```bash
hermes config set memory.provider powermem
```

Put API keys and provider environment variables in `$HERMES_HOME/.env`. Optionally add a full or partial JSON overlay at `$HERMES_HOME/powermem.json` (merged over PowerMem’s env-derived defaults).

## Config (Hermes)

| File | Purpose |
|------|---------|
| `$HERMES_HOME/.env` | Env vars read by PowerMem `auto_config()` |
| `$HERMES_HOME/powermem.json` | Merged over auto-derived config |
| `$HERMES_HOME/powermem-hermes.json` | Optional — e.g. `agent_id` to override default `hermes-{profile}` |

## Tools

| Tool | Description |
|------|-------------|
| `powermem_search` | Hybrid / semantic search over stored memories |
| `powermem_add` | Store text; optional intelligent extraction (`infer`) |
| `powermem_profile` | List memories for the scoped user (broad overview) |

## Identifiers

| Key | Default | Notes |
|-----|---------|--------|
| `user_id` | `hermes-user` | Gateway passes platform user id when available |
| `agent_id` | `hermes-{profile}` | Overridable via `powermem-hermes.json` |

## See also

- User guide: [Memory Providers](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers#powermem) (PowerMem section)
- Upstream: [oceanbase/powermem](https://github.com/oceanbase/powermem)
