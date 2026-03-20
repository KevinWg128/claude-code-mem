# 🧠 Claude Code Memory — MCP Server

Persistent, semantic memory for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) across sessions. Never start from zero again.

An [MCP](https://modelcontextprotocol.io/) server backed by **PostgreSQL + pgvector** that gives Claude Code a long-term memory system — conversations, architectural knowledge, bugs, preferences, and more are stored, embedded, and recalled via semantic search.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Partitioned context** | Recalls relevant history, knowledge, bugs, and preferences in one call |
| **Semantic search** | pgvector cosine similarity across all memory stores |
| **10 memory stores** | Conversations, tool logs, knowledge base, workflows, toolbox, entities, summaries, codebase notes, bugs, preferences |
| **12 MCP tools** | `get_context`, `remember`, `recall`, `log_conversation`, `summarise_thread`, `log_work`, `list_sessions`, `remember_codebase`, `remember_bug`, `remember_preference`, `recall_codebase`, `search_tools` |
| **Flexible embeddings** | Local HuggingFace models (default) or OpenAI API embeddings |
| **Auto tool logging** | Included shell hook logs every Claude Code tool call to the DB automatically |
| **Session summaries** | Two-step summarization compresses conversation history for future recall |

---

## 📦 Prerequisites

- **Python 3.10+**
- **PostgreSQL 14+** with the [pgvector](https://github.com/pgvector/pgvector) extension
- **Claude Code** (installed and working)

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/KevinWg128/claude-code-mem.git
cd claude-code-mem
```

### 2. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `psycopg2-binary` | PostgreSQL driver |
| `pgvector` | pgvector type support for psycopg2 |
| `sentence-transformers` | Local embedding models (default: `paraphrase-mpnet-base-v2`) |
| `mcp[cli]` | MCP server framework |
| `openai` | Optional — only needed if using OpenAI embeddings |

### 4. Set up PostgreSQL

#### a) Create the database and user

```sql
-- Connect as a superuser (e.g. postgres)
CREATE USER agent_mcp WITH PASSWORD 'agent_mcp_password';
CREATE DATABASE agent_memory OWNER agent_mcp;
```

#### b) Enable pgvector

```sql
-- Connect to the agent_memory database
\c agent_memory
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

#### c) Run the schema

```bash
psql -U agent_mcp -d agent_memory -f schema.sql
```

The schema creates 10 tables (2 SQL + 8 vector) with IVFFlat indexes. It's safe to re-run — all statements use `IF NOT EXISTS`.

> [!NOTE]
> The default embedding dimension is **768** (sentence-transformers). If using OpenAI embeddings, edit `schema.sql` to change `vector(768)` to `vector(1536)` (text-embedding-3-small) or `vector(3072)` (text-embedding-3-large) **before** running the schema.

### 5. Configure Claude Code

Copy the example config and edit the paths to match your system:

```bash
cp .mcp.json.example .mcp.json
```

Edit `.mcp.json`:

```json
{
    "mcpServers": {
        "agent-memory": {
            "type": "stdio",
            "command": "/absolute/path/to/claude-code-mem/.venv/bin/python",
            "args": [
                "/absolute/path/to/claude-code-mem/mcp_memory_server.py"
            ],
            "env": {
                "MEMORY_DB_DSN": "postgresql://agent_mcp:agent_mcp_password@localhost:5432/agent_memory",
                "MEMORY_EMBEDDING_MODEL": "sentence-transformers/paraphrase-mpnet-base-v2",
                "MEMORY_DEFAULT_THREAD": "claude-code"
            }
        }
    }
}
```

> [!IMPORTANT]
> Use **absolute paths** for both `command` and `args`. Relative paths will fail when Claude Code launches the server.

Place this file in either:
- **Per-project**: `.mcp.json` in your project root
- **Global**: `~/.claude.json` (applies to all projects)

### 6. Set up CLAUDE.md (recommended)

Copy the example into your project to instruct Claude Code on *when* to use the memory tools:

```bash
cp CLAUDE-example.md CLAUDE.md
```

Edit the `repo` values to match your project name. This file tells Claude Code to:
- Call `get_context()` at session start
- Store bugs, architecture notes, and preferences as they arise
- Summarize the thread at session end

### 7. Set up the auto-logging hook (optional)

The `hooks/post_tool_log.sh` script automatically logs every tool call to the database without LLM involvement.

To enable it, add to your `.claude/settings.json` or project settings:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "command": "/absolute/path/to/claude-code-mem/hooks/post_tool_log.sh"
      }
    ]
  }
}
```

Edit the `HOOK_PYTHON` variable inside `post_tool_log.sh` to point to your venv's Python:

```bash
HOOK_PYTHON="/absolute/path/to/claude-code-mem/.venv/bin/python"
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MEMORY_DB_DSN` | **Yes** | — | PostgreSQL connection string |
| `MEMORY_EMBEDDING_MODEL` | No | `sentence-transformers/paraphrase-mpnet-base-v2` | Embedding model (see below) |
| `MEMORY_DEFAULT_THREAD` | No | `claude-code` | Default conversation thread ID |
| `OPENAI_API_KEY` | Conditional | — | Required only when using `openai/` embedding models |

### Embedding model options

| Model | Dimensions | Speed | Quality |
|---|---|---|---|
| `sentence-transformers/paraphrase-mpnet-base-v2` | 768 | ⚡ Local, no API key | Good |
| `openai/text-embedding-3-small` | 1536 | API call | Better |
| `openai/text-embedding-3-large` | 3072 | API call | Best |

---

## 🔧 MCP Tools Reference

### Core Memory

| Tool | Description |
|---|---|
| `get_context(query, thread_id?, repo?)` | Build a full context prompt from all stores — call at **session start** |
| `remember(store, content, metadata?)` | Write to any memory store |
| `recall(query, store?, k?, repo?)` | Semantic search across any store |

### Conversation

| Tool | Description |
|---|---|
| `log_conversation(content, role?, thread_id?)` | Append a turn to the thread history |
| `summarise_thread(thread_id?, summary_text?, description?)` | Compress thread history into a summary |

### Work Tracking

| Tool | Description |
|---|---|
| `log_work(tool_name, outcome, thread_id?, status?, input_args?, duration_ms?)` | Record a completed task |
| `list_sessions(limit?)` | List recent conversation threads |

### Code-Specific

| Tool | Description |
|---|---|
| `remember_codebase(content, repo, memory_type?, file_path?)` | Store architectural decisions or module descriptions |
| `remember_bug(content, repo?, error_type?, root_cause?, fix_applied?, file_path?, resolved?)` | Record a bug, cause, and fix |
| `remember_preference(content, preference_key?, preference_val?, scope?, scope_value?)` | Store coding/style preferences |
| `recall_codebase(query, repo?, memory_type?, k?)` | Search architectural notes |

### Tool Registry

| Tool | Description |
|---|---|
| `search_tools(query, k?)` | Find tools by natural language description |

---

## 🗄️ Database Schema

The schema creates 10 tables across two categories:

**SQL tables** (exact retrieval):
- `conversational_memory` — threaded chat history
- `tool_log_memory` — audit trail of tool invocations

**Vector tables** (semantic search via pgvector):
- `semantic_memory` — knowledge base
- `workflow_memory` — procedural patterns
- `toolbox_memory` — semantic tool registry
- `entity_memory` — named entities
- `summary_memory` — compressed session summaries
- `codebase_memory` — architectural decisions (filtered by repo)
- `bug_memory` — bug/fix history (filtered by repo)
- `preference_memory` — user/project preferences (scoped)

---

## 🧪 Verify the Setup

1. **Check the MCP server starts:**

   ```bash
   source .venv/bin/activate
   MEMORY_DB_DSN="postgresql://agent_mcp:agent_mcp_password@localhost:5432/agent_memory" \
   python mcp_memory_server.py
   ```

   You should see log lines on stderr. Press `Ctrl+C` to stop.

2. **Test from Claude Code:**

   Open Claude Code in a project with the `.mcp.json` configured. Run:

   ```
   /mcp
   ```

   You should see `agent-memory` listed with 12 tools.

3. **Run the MCP Inspector** (optional):

   ```bash
   npx @modelcontextprotocol/inspector \
     .venv/bin/python mcp_memory_server.py
   ```

---

## 📁 Project Structure

```
claude-code-mem/
├── mcp_memory_server.py   # MCP server — 12 tools exposed via stdio
├── memory_manager.py      # Core logic — PostgreSQL + pgvector read/write
├── schema.sql             # Database schema (safe to re-run)
├── requirements.txt       # Python dependencies
├── .mcp.json.example      # Example MCP config for Claude Code
├── CLAUDE-example.md      # Example CLAUDE.md with memory instructions
├── hooks/
│   └── post_tool_log.sh   # Auto-logging hook for tool calls
└── LICENSE                # MIT
```

---

## 📄 License

[MIT](LICENSE) © KevinWg128
