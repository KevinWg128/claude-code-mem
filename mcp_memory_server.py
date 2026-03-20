"""
mcp_memory_server.py
====================
MCP server that gives Claude Code persistent, semantic memory across sessions.

Exposes 12 tools backed by MemoryManager (PostgreSQL + pgvector):

  Core memory
  ───────────
  get_context          Build a full partitioned context prompt for a session
  remember             Write to any memory store
  recall               Semantic search across any store

  Conversation
  ────────────
  log_conversation     Append a user/assistant turn to a thread
  summarise_thread     Compress a thread's history into a summary

  Work log
  ────────
  log_work             Record a completed task and its outcome
  list_sessions        List recent conversation threads

  Code-specific (the three new stores)
  ────────────────────────────────────
  remember_codebase    Store an architectural decision or module description
  remember_bug         Record a bug, root cause, and fix
  remember_preference  Store a coding/style preference
  recall_codebase      Semantic search over codebase decisions

  Tool registry
  ─────────────
  register_tool        Store a tool definition for semantic retrieval
  search_tools         Find relevant tools by natural language description

Transport: stdio  (the only transport Claude Code uses for local servers)

Configuration (add to ~/.claude.json or .mcp.json):
  {
    "mcpServers": {
      "agent-memory": {
        "command": "python",
        "args": ["/absolute/path/to/mcp_memory_server.py"],
        "env": {
          "MEMORY_DB_DSN":       "postgresql://user:pass@localhost:5432/agent_memory",
          "MEMORY_EMBEDDING_MODEL": "sentence-transformers/paraphrase-mpnet-base-v2",
          "MEMORY_DEFAULT_THREAD": "claude-code"
        }
      }
    }
  }

Required env vars:
  MEMORY_DB_DSN              Full psycopg2 DSN string
  MEMORY_EMBEDDING_MODEL     HuggingFace model name  (default: paraphrase-mpnet-base-v2)
                             OR "openai/text-embedding-3-small" / "openai/text-embedding-3-large"
  MEMORY_DEFAULT_THREAD      Thread id used when none is specified  (default: "claude-code")
  OPENAI_API_KEY             Required only if using OpenAI embeddings or the entity extraction path
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: make memory_manager importable from the same directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from memory_manager import MemoryManager, connect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_memory_server")

# ---------------------------------------------------------------------------
# Env / config
# ---------------------------------------------------------------------------

DB_DSN = os.environ.get("MEMORY_DB_DSN", "")
EMBEDDING_MODEL_NAME = os.environ.get(
    "MEMORY_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-mpnet-base-v2",
)
DEFAULT_THREAD = os.environ.get("MEMORY_DEFAULT_THREAD", "claude-code")

# ---------------------------------------------------------------------------
# Embedding model factory
# ---------------------------------------------------------------------------

def _build_embedding_model(name: str):
    """
    Build an embedding model from a name string.

    Supported:
      "sentence-transformers/..."         → local HuggingFace model
      "openai/text-embedding-3-small"     → OpenAI API (1536 dims)
      "openai/text-embedding-3-large"     → OpenAI API (3072 dims)
    """
    if name.startswith("openai/"):
        model_id = name.split("/", 1)[1]
        from openai import OpenAI

        client = OpenAI()

        class _OpenAIEmbedder:
            def __init__(self, model_id):
                self._id = model_id

            def embed_query(self, text: str) -> list[float]:
                resp = client.embeddings.create(input=[text], model=self._id)
                return resp.data[0].embedding

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                resp = client.embeddings.create(input=texts, model=self._id)
                return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

        return _OpenAIEmbedder(model_id)

    # Local HuggingFace model
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=name)
    except ImportError:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer(name)

        class _STEmbedder:
            def embed_query(self, text: str) -> list[float]:
                return model.encode([text], normalize_embeddings=True)[0].tolist()

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return model.encode(texts, normalize_embeddings=True).tolist()

        return _STEmbedder()


# ---------------------------------------------------------------------------
# Lazy-init singletons (initialised on first tool call, not at import time,
# so Claude Code can load the server definition before the DB is up)
# ---------------------------------------------------------------------------

_conn = None
_mm: MemoryManager | None = None


def _get_mm() -> MemoryManager:
    global _conn, _mm
    if _mm is None:
        if not DB_DSN:
            raise RuntimeError(
                "MEMORY_DB_DSN environment variable is not set. "
                "Set it to your PostgreSQL connection string."
            )
        logger.info("Connecting to %s", DB_DSN.split("@")[-1])
        _conn = connect(DB_DSN)
        em = _build_embedding_model(EMBEDDING_MODEL_NAME)
        _mm = MemoryManager(_conn, em)
        logger.info("MemoryManager initialised (model=%s)", EMBEDDING_MODEL_NAME)
    return _mm


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="agent-memory",
    instructions=textwrap.dedent("""
        Persistent memory server for Claude Code.

        Use these tools to maintain context across sessions:

        SESSION START
          1. Call get_context(query=<your task>, repo=<repo name if known>)
             to warm up with all relevant memory.

        DURING WORK
          2. Call log_conversation(role="user"|"assistant", content=...) for
             each significant exchange.
          3. Call remember(store="knowledge_base", content=...) when you
             discover a fact worth keeping.
          4. Call remember_codebase(content=..., repo=...) when you learn
             something architectural (design decisions, rejected patterns,
             module purposes).
          5. Call remember_bug(content=..., repo=...) when you fix a bug.
          6. Call log_work(tool_name=..., outcome=...) after each tool use.

        SESSION END
          7. Call summarise_thread(thread_id=...) before ending to compress
             the conversation for future recall.
          8. Call remember_preference(content=...) for any style or
             preference you learned about this user/project.
    """).strip(),
)


# ============================================================================
# Tool 1: get_context
# ============================================================================

@mcp.tool()
def get_context(
    query: str,
    thread_id: str = DEFAULT_THREAD,
    repo: str = "",
) -> str:
    """
    Build a full partitioned context window from all memory stores.

    Call this at the START of every Claude Code session. It retrieves:
    - Recent conversation history for the thread
    - Semantically relevant knowledge base entries
    - Relevant workflow patterns used before
    - Named entities related to the query
    - Any available summaries (with expand references)
    - Codebase architectural notes (if repo is given)
    - Known bugs for this repo (if repo is given)
    - User/project preferences

    Returns a structured prompt string ready to be prepended to the
    system prompt or injected as context.

    Args:
        query:     The current task or question (used for semantic retrieval).
        thread_id: Conversation thread id (default: env MEMORY_DEFAULT_THREAD).
        repo:      Short repo name or git remote (enables codebase + bug recall).
    """
    mm = _get_mm()
    return mm.build_context(query, thread_id, repo=repo or None)


# ============================================================================
# Tool 2: remember
# ============================================================================

STORE_CHOICES = (
    "knowledge_base",
    "workflow",
    "entity",
    "summary",
    "codebase",
    "bug",
    "preference",
)


@mcp.tool()
def remember(
    store: str,
    content: str,
    metadata: str = "{}",
) -> str:
    """
    Write a piece of information to a memory store.

    Use this to persist anything worth keeping: facts discovered during
    research, patterns that worked, entities referenced, etc.

    Args:
        store:    One of: knowledge_base, workflow, entity, summary,
                  codebase, bug, preference.
        content:  The text to embed and store.
        metadata: Optional JSON string of key/value tags, e.g.
                  '{"source": "arxiv", "title": "MemGPT"}'.

    Returns:
        The UUID of the new memory row.
    """
    mm = _get_mm()
    meta = _parse_json(metadata, {})

    dispatch = {
        "knowledge_base": lambda: mm.write_knowledge_base(content, meta),
        "workflow":       lambda: mm.write_workflow(content, meta),
        "entity":         lambda: mm.write_entity(
                              meta.get("name", content.split(":")[0].strip()),
                              meta.get("entity_type", "unknown"),
                              content,
                          ),
        "summary":        lambda: _remember_summary(mm, content, meta),
        "codebase":       lambda: mm.write_codebase(
                              content,
                              repo=meta.get("repo", "unknown"),
                              memory_type=meta.get("memory_type", "architecture"),
                              file_path=meta.get("file_path"),
                              metadata=meta,
                          ),
        "bug":            lambda: mm.write_bug(
                              content,
                              repo=meta.get("repo"),
                              error_type=meta.get("error_type"),
                              root_cause=meta.get("root_cause"),
                              fix_applied=meta.get("fix_applied"),
                          ),
        "preference":     lambda: mm.write_preference(
                              content,
                              scope=meta.get("scope", "global"),
                              scope_value=meta.get("scope_value"),
                              preference_key=meta.get("preference_key"),
                              preference_val=meta.get("preference_val"),
                          ),
    }
    if store not in dispatch:
        return f"Unknown store '{store}'. Choose from: {', '.join(STORE_CHOICES)}"

    result = dispatch[store]()
    row_id = result[0] if isinstance(result, list) else result
    return f"Stored in {store} (id={row_id})"


def _remember_summary(mm: MemoryManager, content: str, meta: dict) -> str:
    import uuid
    sid = meta.get("id") or str(uuid.uuid4())[:8]
    mm.write_summary(
        sid, content, content,
        description=meta.get("description", "summary"),
        thread_id=meta.get("thread_id"),
    )
    return sid


# ============================================================================
# Tool 3: recall
# ============================================================================

@mcp.tool()
def recall(
    query: str,
    store: str = "knowledge_base",
    k: int = 5,
    repo: str = "",
) -> str:
    """
    Semantic similarity search over a memory store.

    Finds the k most relevant memories for a natural language query.
    Use this to look up facts, patterns, or past context on demand.

    Args:
        query: Natural language search query.
        store: Which store to search. One of: knowledge_base, workflow,
               entity, summary, toolbox, codebase, bug, preference.
        k:     Number of results to return (default: 5).
        repo:  Filter codebase/bug results to a specific repo.

    Returns:
        Formatted string of matching results with similarity scores.
    """
    mm = _get_mm()
    dispatch = {
        "knowledge_base": lambda: mm.read_knowledge_base(query, k=k),
        "workflow":       lambda: mm.read_workflow(query, k=k),
        "entity":         lambda: mm.read_entity(query, k=k),
        "summary":        lambda: mm.read_summary_context(query, k=k),
        "toolbox":        lambda: _format_tools(mm.read_toolbox(query, k=k)),
        "codebase":       lambda: mm.read_codebase(query, repo=repo or None, k=k),
        "bug":            lambda: mm.read_bug(query, repo=repo or None, k=k),
        "preference":     lambda: mm.read_preference(query, k=k),
    }
    if store not in dispatch:
        return f"Unknown store '{store}'. Choose from: {', '.join(dispatch)}"
    return dispatch[store]()


def _format_tools(tools: list[dict]) -> str:
    if not tools:
        return "## Toolbox Memory\n(no matching tools)\n"
    lines = [f"- {t['function']['name']}: {t['function']['description'][:120]}"
             for t in tools]
    return "## Toolbox Memory\n" + "\n".join(lines) + "\n"


# ============================================================================
# Tool 4: log_conversation
# ============================================================================

@mcp.tool()
def log_conversation(
    content: str,
    role: str = "user",
    thread_id: str = DEFAULT_THREAD,
) -> str:
    """
    Append a conversation turn to the persistent thread history.

    Call this after each significant user message and assistant response
    so the thread can be recalled and summarised in future sessions.

    Args:
        content:   The message text.
        role:      "user", "assistant", "system", or "tool".
        thread_id: Conversation thread id (default: env MEMORY_DEFAULT_THREAD).

    Returns:
        The new row UUID.
    """
    mm = _get_mm()
    row_id = mm.write_conversational_memory(content, role, thread_id)
    return f"Logged (id={row_id}, thread={thread_id}, role={role})"


# ============================================================================
# Tool 5: summarise_thread
# ============================================================================

@mcp.tool()
def summarise_thread(
    thread_id: str = DEFAULT_THREAD,
    summary_text: str = "",
    description: str = "",
) -> str:
    """
    Compress a thread's unsummarized conversation history into a summary.

    Call this at the END of a session (or when context grows large) to
    archive the conversation. The summary is stored in summary_memory and
    the source rows are tagged with summary_id so they are excluded from
    future read_conversational_memory() calls.

    **Two-step usage (Claude Code generates the summary):**

    Step 1 — Get the transcript:
      Call summarise_thread(thread_id="...") with no summary_text.
      The tool returns the raw transcript. Use it to write a summary.

    Step 2 — Store the summary:
      Call summarise_thread(thread_id="...", summary_text="...", description="...")
      with the summary you generated. The tool stores it and marks
      the source messages as summarized.

    Your summary should use these headings:
      ### Work Done
      ### Decisions Made
      ### Open Items
      ### Entities & References

    Keep concrete details (file names, error types, API names, decisions).
    Be concise. Do not invent.

    The description should be an 8-12 word label for the summary.

    Args:
        thread_id:    The thread to summarise.
        summary_text: The summary YOU generated (empty = return transcript).
        description:  Short 8-12 word label for the summary.

    Returns:
        If summary_text is empty: the raw transcript to summarize.
        If summary_text is provided: summary ID and description label.
    """
    import uuid

    mm = _get_mm()

    rows = mm.read_unsummarized_rows(thread_id)
    if not rows:
        return f"Nothing to summarise in thread '{thread_id}'."

    message_ids = [r[0] for r in rows]
    transcript = "\n".join(
        f"[{r[3].strftime('%Y-%m-%d %H:%M:%S')}] [{r[1].upper()}] {r[2]}"
        for r in rows
    )

    # Step 1: If no summary provided, return the transcript for Claude Code
    if not summary_text.strip():
        return (
            f"Thread '{thread_id}' has {len(message_ids)} unsummarized messages.\n\n"
            f"--- TRANSCRIPT ---\n{transcript[:8000]}\n--- END ---\n\n"
            "Now generate a summary using the headings: "
            "Work Done, Decisions Made, Open Items, Entities & References. "
            "Then call summarise_thread again with summary_text and description."
        )

    # Step 2: Store the provided summary
    if not description.strip():
        description = "Session summary"

    summary_id = str(uuid.uuid4())[:8]
    mm.write_summary(summary_id, transcript, summary_text, description,
                     thread_id=thread_id)
    mm.mark_conversations_summarized(message_ids, summary_id)

    return (
        f"Summarised {len(message_ids)} messages → "
        f"[Summary ID: {summary_id}] {description}"
    )


# ============================================================================
# Tool 6: log_work
# ============================================================================

@mcp.tool()
def log_work(
    tool_name: str,
    outcome: str,
    thread_id: str = DEFAULT_THREAD,
    status: str = "success",
    input_args: str = "{}",
    duration_ms: int = 0,
) -> str:
    """
    Record a completed tool invocation or task to the audit trail.

    Call this after every significant tool use or task completion so
    the agent has a searchable history of what was done and what worked.

    Args:
        tool_name:  Name of the tool or task (e.g. "bash", "edit_file").
        outcome:    Human-readable description of what happened / was produced.
        thread_id:  Current session thread.
        status:     "success", "error", or "timeout".
        input_args: JSON string of the inputs used (optional).
        duration_ms: Execution time in milliseconds (optional).

    Returns:
        The new log row UUID.
    """
    mm = _get_mm()
    args = _parse_json(input_args, {})
    row_id = mm.write_tool_log(
        tool_name=tool_name,
        input_args=args,
        output=outcome,
        status=status,
        thread_id=thread_id,
        duration_ms=duration_ms if duration_ms else None,
    )
    return f"Logged tool use (id={row_id}, tool={tool_name}, status={status})"


# ============================================================================
# Tool 7: list_sessions
# ============================================================================

@mcp.tool()
def list_sessions(limit: int = 10) -> str:
    """
    List recent conversation threads with their message counts and timestamps.

    Use this to browse past work sessions and find relevant thread IDs
    to pass to get_context() or summarise_thread().

    Args:
        limit: Maximum number of threads to return (default: 10).

    Returns:
        A formatted list of threads with last-active timestamps.
    """
    mm = _get_mm()
    with mm.conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT thread_id,
                   COUNT(*) AS messages,
                   MAX(created_at) AS last_active,
                   COUNT(*) FILTER (WHERE summary_id IS NULL) AS unsummarized
            FROM   {mm.conversation_table}
            GROUP  BY thread_id
            ORDER  BY last_active DESC
            LIMIT  %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

    if not rows:
        return "No conversation sessions found."

    lines = ["Thread ID".ljust(36) + "  Msgs  Unsummarized  Last Active"]
    lines.append("─" * 80)
    for thread_id, msgs, last_active, unsummarized in rows:
        ts = last_active.strftime("%Y-%m-%d %H:%M") if last_active else "—"
        lines.append(
            f"{str(thread_id):<36}  {msgs:>4}  {unsummarized:>12}  {ts}"
        )
    return "\n".join(lines)


# ============================================================================
# Tool 8: remember_codebase
# ============================================================================

@mcp.tool()
def remember_codebase(
    content: str,
    repo: str,
    memory_type: str = "architecture",
    file_path: str = "",
) -> str:
    """
    Store a code-specific architectural note, convention, or design decision.

    Use this when you learn something structural about a codebase that
    should persist across sessions: design decisions, rejected patterns,
    module purposes, dependency rationale, or coding conventions.

    Args:
        content:     The note to store (be specific — this is embedded and
                     retrieved by semantic search).
        repo:        Short repo name or git remote URL.
        memory_type: One of: architecture, module, convention, dependency,
                     constraint  (default: architecture).
        file_path:   Relevant file or directory, if applicable.

    Returns:
        The new row UUID.
    """
    valid_types = {"architecture", "module", "convention", "dependency", "constraint"}
    if memory_type not in valid_types:
        memory_type = "architecture"

    mm = _get_mm()
    row_id = mm.write_codebase(
        content, repo=repo, memory_type=memory_type,
        file_path=file_path or None,
    )
    return f"Stored codebase note (id={row_id}, repo={repo}, type={memory_type})"


# ============================================================================
# Tool 9: remember_bug
# ============================================================================

@mcp.tool()
def remember_bug(
    content: str,
    repo: str = "",
    error_type: str = "",
    root_cause: str = "",
    fix_applied: str = "",
    file_path: str = "",
    resolved: bool = True,
) -> str:
    """
    Record a bug, its root cause, and the fix applied.

    Future sessions can recall similar bugs by semantic similarity, short-
    circuiting debugging when the same class of problem recurs.

    The content field should be a self-contained description:
      "TypeError: NoneType is not subscriptable in auth.py middleware.
       Root cause: JWT decode returns None when token is expired instead
       of raising. Fix: check return value before indexing."

    Args:
        content:    Full description (bug + cause + fix in plain English).
        repo:       Repo name (used for filtered recall).
        error_type: Exception class or category (e.g. "asyncio.TimeoutError").
        root_cause: One-line root cause summary.
        fix_applied: One-line description of the fix.
        file_path:  File where the bug was found.
        resolved:   Whether this bug is fixed (default: True).

    Returns:
        The new row UUID.
    """
    mm = _get_mm()
    row_id = mm.write_bug(
        content,
        repo=repo or None,
        error_type=error_type or None,
        root_cause=root_cause or None,
        fix_applied=fix_applied or None,
        file_path=file_path or None,
        resolved=resolved,
    )
    return f"Stored bug record (id={row_id})"


# ============================================================================
# Tool 10: remember_preference
# ============================================================================

@mcp.tool()
def remember_preference(
    content: str,
    preference_key: str = "",
    preference_val: str = "",
    scope: str = "global",
    scope_value: str = "",
) -> str:
    """
    Store a user or project preference.

    Use this when you observe a consistent preference that should apply
    to future sessions: test framework choice, import style, comment
    verbosity, preferred libraries, naming conventions, etc.

    Examples:
      remember_preference("Use pytest, not unittest, for all tests",
                          preference_key="test_framework",
                          preference_val="pytest")

      remember_preference("Add type hints to all new functions",
                          scope="repo", scope_value="my-api")

    Args:
        content:        Human-readable description of the preference.
        preference_key: Short slug key (e.g. "test_framework"). If set,
                        calling this again with the same key upserts.
        preference_val: The preference value (e.g. "pytest").
        scope:          "global" | "repo" | "language" | "session".
        scope_value:    The repo name or language this applies to.

    Returns:
        The new row UUID.
    """
    mm = _get_mm()
    row_id = mm.write_preference(
        content,
        scope=scope,
        scope_value=scope_value or None,
        preference_key=preference_key or None,
        preference_val=preference_val or None,
    )
    return f"Stored preference (id={row_id}, key={preference_key or '—'})"


# ============================================================================
# Tool 11: recall_codebase
# ============================================================================

@mcp.tool()
def recall_codebase(
    query: str,
    repo: str = "",
    memory_type: str = "",
    k: int = 5,
) -> str:
    """
    Retrieve architectural notes and decisions relevant to the current task.

    Use this when starting work in a familiar repo to warm up with
    structural context: what patterns are used, what is forbidden,
    what each module does, which libraries were chosen and why.

    Args:
        query:       Natural language description of what you're working on.
        repo:        Filter to a specific repo (optional).
        memory_type: Filter by type: architecture, module, convention,
                     dependency, or constraint (optional).
        k:           Number of results (default: 5).

    Returns:
        Formatted list of matching architectural notes.
    """
    mm = _get_mm()
    return mm.read_codebase(
        query,
        repo=repo or None,
        memory_type=memory_type or None,
        k=k,
    )


# ============================================================================
# Tool 12: search_tools
# ============================================================================

@mcp.tool()
def search_tools(query: str, k: int = 5) -> str:
    """
    Find registered tools semantically relevant to a task or problem.

    Use this when you need a capability and are not sure which tool to use,
    or to discover what tools are available for a specific type of problem.

    Args:
        query: Natural language description of the task or problem.
        k:     Number of tools to return (default: 5).

    Returns:
        List of matching tool names, descriptions, and parameter schemas.
    """
    mm = _get_mm()
    tools = mm.read_toolbox(query, k=k)
    if not tools:
        return "No matching tools found in the toolbox."

    lines = []
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        param_str = ", ".join(
            f"{k}: {v.get('type', '?')}" for k, v in params.items()
        )
        lines.append(f"• {fn['name']}({param_str})\n  {fn['description'][:200]}")

    return "## Matching Tools\n\n" + "\n\n".join(lines)


# ============================================================================
# Utility
# ============================================================================

def _parse_json(raw: str, default: Any) -> Any:
    """Safely parse a JSON string; return default on failure."""
    if not raw or raw.strip() in ("{}", ""):
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting agent-memory MCP server (stdio transport)")
    logger.info("DB:    %s", DB_DSN.split("@")[-1] if "@" in DB_DSN else "(not set)")
    logger.info("Model: %s", EMBEDDING_MODEL_NAME)
    logger.info("Thread default: %s", DEFAULT_THREAD)
    mcp.run(transport="stdio")