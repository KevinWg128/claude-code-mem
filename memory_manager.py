"""
memory_manager.py
=================
PostgreSQL + pgvector drop-in replacement for the Oracle-backed memory
architecture built in Labs L2–L5.

Replaces:
  langchain_oracledb.vectorstores.OracleVS  →  direct psycopg2 + pgvector
  oracledb connection                        →  psycopg2 connection

Dependencies:
  pip install psycopg2-binary pgvector sentence-transformers openai

Tables expected (see schema.sql):
  conversational_memory   – SQL, exact retrieval
  tool_log_memory         – SQL, audit trail
  semantic_memory         – vector, knowledge base
  workflow_memory         – vector, procedural patterns
  toolbox_memory          – vector, semantic tool registry
  entity_memory           – vector, named entities
  summary_memory          – vector, compressed context
  codebase_memory         – vector, code-specific architecture
  bug_memory              – vector, bug/fix history
  preference_memory       – vector, user/project preferences
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def connect(
    dsn: str | None = None,
    *,
    host: str = "localhost",
    port: int = 5432,
    dbname: str = "agent_memory",
    user: str = "postgres",
    password: str = "",
) -> psycopg2.extensions.connection:
    """
    Open a psycopg2 connection and register the pgvector type extension.

    Prefer passing a full DSN string:
        connect("postgresql://user:pass@host:5432/dbname")

    Or use keyword arguments for individual params.
    """
    conn = psycopg2.connect(
        dsn or f"host={host} port={port} dbname={dbname} user={user} password={password}"
    )
    conn.autocommit = False
    register_vector(conn)
    psycopg2.extras.register_uuid(conn)
    return conn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _embed(texts: list[str], embedding_model) -> list[list[float]]:
    """Embed a list of strings; return list of float vectors."""
    if hasattr(embedding_model, "embed_documents"):
        # LangChain Embeddings interface
        return embedding_model.embed_documents(texts)
    # Direct sentence-transformers model
    return embedding_model.encode(texts, normalize_embeddings=True).tolist()


def _embed_query(text: str, embedding_model) -> list[float]:
    """Embed a single query string."""
    if hasattr(embedding_model, "embed_query"):
        return embedding_model.embed_query(text)
    return embedding_model.encode([text], normalize_embeddings=True)[0].tolist()


def _fmt_rows(rows: list[tuple], headers: list[str]) -> str:
    """Format a list of DB rows as a readable string block."""
    if not rows:
        return ""
    lines = []
    for row in rows:
        parts = [f"{h}: {v}" for h, v in zip(headers, row) if v is not None]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# StoreManager  (mirrors the Oracle StoreManager from helper.py)
# ---------------------------------------------------------------------------

class StoreManager:
    """
    Thin initialisation wrapper that mirrors the Oracle StoreManager.
    Validates the connection and stores references consumed by MemoryManager.
    """

    def __init__(
        self,
        conn: psycopg2.extensions.connection,
        embedding_model,
        *,
        conversational_table: str = "conversational_memory",
        tool_log_table: str = "tool_log_memory",
        knowledge_base_table: str = "semantic_memory",
        workflow_table: str = "workflow_memory",
        toolbox_table: str = "toolbox_memory",
        entity_table: str = "entity_memory",
        summary_table: str = "summary_memory",
        codebase_table: str = "codebase_memory",
        bug_table: str = "bug_memory",
        preference_table: str = "preference_memory",
    ) -> None:
        self.conn = conn
        self.embedding_model = embedding_model
        self.conversational_table = conversational_table
        self.tool_log_table = tool_log_table
        self.knowledge_base_table = knowledge_base_table
        self.workflow_table = workflow_table
        self.toolbox_table = toolbox_table
        self.entity_table = entity_table
        self.summary_table = summary_table
        self.codebase_table = codebase_table
        self.bug_table = bug_table
        self.preference_table = preference_table

    # -- Getters (lab compatibility) -----------------------------------------

    def get_conn(self) -> psycopg2.extensions.connection:
        return self.conn

    def get_conversational_table(self) -> str:
        return self.conversational_table

    def get_tool_log_table(self) -> str:
        return self.tool_log_table

    def get_knowledge_base_table(self) -> str:
        return self.knowledge_base_table

    def get_workflow_table(self) -> str:
        return self.workflow_table

    def get_toolbox_table(self) -> str:
        return self.toolbox_table

    def get_entity_table(self) -> str:
        return self.entity_table

    def get_summary_table(self) -> str:
        return self.summary_table


# ---------------------------------------------------------------------------
# MemoryManager  (core class, mirrors helper.py MemoryManager)
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Unified read/write interface over all ten memory stores.

    SQL stores  → conversational_memory, tool_log_memory
    Vector stores → semantic_memory, workflow_memory, toolbox_memory,
                    entity_memory, summary_memory,
                    codebase_memory, bug_memory, preference_memory

    API is intentionally identical to the Oracle version in helper.py so
    existing agent loop code (L5 call_agent) works without changes.
    """

    def __init__(
        self,
        conn: psycopg2.extensions.connection,
        embedding_model,
        *,
        conversation_table: str = "conversational_memory",
        tool_log_table: str = "tool_log_memory",
        knowledge_base_table: str = "semantic_memory",
        workflow_table: str = "workflow_memory",
        toolbox_table: str = "toolbox_memory",
        entity_table: str = "entity_memory",
        summary_table: str = "summary_memory",
        codebase_table: str = "codebase_memory",
        bug_table: str = "bug_memory",
        preference_table: str = "preference_memory",
    ) -> None:
        self.conn = conn
        self.embedding_model = embedding_model
        # Table name attributes (accessed directly by summarize_conversation etc.)
        self.conversation_table = conversation_table
        self.tool_log_table = tool_log_table
        self.knowledge_base_table = knowledge_base_table
        self.workflow_table = workflow_table
        self.toolbox_table = toolbox_table
        self.entity_table = entity_table
        self.summary_table = summary_table
        self.codebase_table = codebase_table
        self.bug_table = bug_table
        self.preference_table = preference_table

    # =========================================================================
    # Internal: vector insert / search
    # =========================================================================

    def _insert_vector(
        self,
        table: str,
        content: str,
        metadata: dict | None,
        extra_cols: dict | None = None,
    ) -> str:
        """
        Insert one row into a vector table.
        Returns the new row's id (UUID string).
        """
        embedding = _embed_query(content, self.embedding_model)
        row_id = str(uuid.uuid4())
        cols = ["id", "content", "embedding", "metadata", "created_at"]
        vals = [row_id, content, embedding, json.dumps(metadata or {}), _now_utc()]

        if extra_cols:
            for k, v in extra_cols.items():
                cols.append(k)
                vals.append(v)

        placeholders = ", ".join(["%s"] * len(vals))
        col_list = ", ".join(cols)
        sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
        with self.conn.cursor() as cur:
            cur.execute(sql, vals)
        self.conn.commit()
        return row_id

    def _insert_vectors_batch(
        self,
        table: str,
        texts: list[str],
        metadatas: list[dict] | None,
    ) -> list[str]:
        """
        Batch insert into a vector table. Returns list of new UUIDs.
        """
        if not texts:
            return []
        embeddings = _embed(texts, self.embedding_model)
        metadatas = metadatas or [{}] * len(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        now = _now_utc()
        rows = [
            (ids[i], texts[i], emb, json.dumps(metadatas[i]), now)
            for i, emb in enumerate(embeddings)
        ]
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {table} (id, content, embedding, metadata, created_at)"
                " VALUES %s",
                rows,
            )
        self.conn.commit()
        return ids

    def _search_vector(
        self,
        table: str,
        query: str,
        k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Cosine ANN search. Returns list of dicts with id, content, metadata, distance.
        Optionally filter on top-level metadata keys (exact match).
        """
        query_vec = _embed_query(query, self.embedding_model)
        where_clauses = []
        filter_params: list[Any] = []

        if metadata_filter:
            for key, val in metadata_filter.items():
                where_clauses.append(f"metadata->>{repr(key)} = %s")
                filter_params.append(str(val))

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        sql = f"""
            SELECT id, content, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM   {table}
            {where_sql}
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s
        """
        # Params must match placeholder order:
        #   %s (SELECT similarity), [WHERE filters...], %s (ORDER BY), %s (LIMIT)
        params: list[Any] = [query_vec] + filter_params + [query_vec, k]

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # 1. CONVERSATIONAL MEMORY  (SQL)
    # =========================================================================

    def write_conversational_memory(
        self,
        content: str,
        role: str,
        thread_id: str,
        metadata: dict | None = None,
    ) -> str:
        """Append one turn to the conversation history. Returns new row id."""
        row_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.conversation_table}
                    (id, thread_id, role, content, created_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (row_id, str(thread_id), role, content, _now_utc(),
                 json.dumps(metadata or {})),
            )
        self.conn.commit()
        return row_id

    def read_conversational_memory(
        self,
        thread_id: str,
        limit: int = 20,
    ) -> str:
        """
        Return the most recent `limit` *unsummarized* turns for a thread,
        formatted as a labelled string block for context injection.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT role, content, created_at
                FROM   {self.conversation_table}
                WHERE  thread_id = %s AND summary_id IS NULL
                ORDER  BY created_at DESC
                LIMIT  %s
                """,
                (str(thread_id), limit),
            )
            rows = cur.fetchall()

        if not rows:
            return "## Conversation Memory\n(no history)\n"

        # Return in chronological order
        lines = [f"[{r[2].strftime('%H:%M:%S')}] [{r[0].upper()}] {r[1]}"
                 for r in reversed(rows)]
        return "## Conversation Memory\n" + "\n".join(lines) + "\n"

    def read_conversations_by_summary_id(self, summary_id: str) -> str:
        """
        Return all conversation rows that were compressed into `summary_id`.
        Used by expand_summary() to walk back to source messages.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT role, content, created_at
                FROM   {self.conversation_table}
                WHERE  summary_id = %s
                ORDER  BY created_at ASC
                """,
                (str(summary_id),),
            )
            rows = cur.fetchall()

        if not rows:
            return "(no source messages found for this summary)"

        lines = [f"[{r[2].strftime('%Y-%m-%d %H:%M:%S')}] [{r[0].upper()}] {r[1]}"
                 for r in rows]
        return "## Original Conversation\n" + "\n".join(lines) + "\n"

    def mark_conversations_summarized(
        self, message_ids: list[str], summary_id: str
    ) -> int:
        """
        Tag a batch of conversation rows with summary_id so they are
        excluded from future read_conversational_memory() calls.
        Returns count of rows updated.
        """
        if not message_ids:
            return 0
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""
                UPDATE {self.conversation_table}
                SET    summary_id = %s
                WHERE  id = %s AND summary_id IS NULL
                """,
                [(summary_id, mid) for mid in message_ids],
                template="(%s, %s)",
            )
            count = cur.rowcount
        self.conn.commit()
        return count

    def read_unsummarized_rows(self, thread_id: str) -> list[tuple]:
        """
        Return (id, role, content, created_at) for all unsummarized rows
        in a thread, ordered chronologically. Used by summarize_conversation().
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, role, content, created_at
                FROM   {self.conversation_table}
                WHERE  thread_id = %s AND summary_id IS NULL
                ORDER  BY created_at ASC
                """,
                (str(thread_id),),
            )
            return cur.fetchall()

    # =========================================================================
    # 2. TOOL LOG MEMORY  (SQL)
    # =========================================================================

    def write_tool_log(
        self,
        tool_name: str,
        input_args: dict | None = None,
        output: str | None = None,
        status: str = "success",
        thread_id: str | None = None,
        duration_ms: int | None = None,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Append one tool invocation record. Returns new row id."""
        row_id = str(uuid.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.tool_log_table}
                    (id, thread_id, tool_name, input_args, output,
                     status, error_message, duration_ms, created_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    row_id, thread_id, tool_name,
                    json.dumps(input_args or {}), output,
                    status, error_message, duration_ms, _now_utc(),
                    json.dumps(metadata or {}),
                ),
            )
        self.conn.commit()
        return row_id

    def read_tool_logs(
        self,
        thread_id: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Return tool log rows as dicts. Filter by thread_id, tool_name,
        or status (any combination).
        """
        clauses, params = [], []
        if thread_id is not None:
            clauses.append("thread_id = %s"); params.append(thread_id)
        if tool_name is not None:
            clauses.append("tool_name = %s"); params.append(tool_name)
        if status is not None:
            clauses.append("status = %s"); params.append(status)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)

        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id, thread_id, tool_name, input_args, output,
                       status, error_message, duration_ms, created_at
                FROM   {self.tool_log_table}
                {where}
                ORDER  BY created_at DESC
                LIMIT  %s
                """,
                params,
            )
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # 3. KNOWLEDGE BASE / SEMANTIC MEMORY  (vector)
    # =========================================================================

    def write_knowledge_base(
        self,
        text: str | list[str],
        metadata: dict | list[dict] | None = None,
    ) -> list[str]:
        """
        Write one document or a batch of document chunks.

        Single:  write_knowledge_base("some text", {"source": "arxiv"})
        Batch:   write_knowledge_base(["chunk1", "chunk2"], [meta1, meta2])
        """
        if isinstance(text, str):
            return [self._insert_vector(self.knowledge_base_table, text, metadata)]
        return self._insert_vectors_batch(self.knowledge_base_table, text, metadata)

    def read_knowledge_base(self, query: str, k: int = 5) -> str:
        """Semantic search; returns formatted string block for context injection."""
        rows = self._search_vector(self.knowledge_base_table, query, k=k)
        if not rows:
            return "## Knowledge Base Memory\n(no relevant documents)\n"

        lines = []
        for r in rows:
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            title = meta.get("title", "")
            src = meta.get("source_type") or meta.get("source", "")
            tag = f" [{title}]" if title else f" [{src}]" if src else ""
            lines.append(f"[sim={r['similarity']:.2f}]{tag} {r['content']}")

        return "## Knowledge Base Memory\n" + "\n\n".join(lines) + "\n"

    # =========================================================================
    # 4. WORKFLOW MEMORY  (vector)
    # =========================================================================

    def write_workflow(self, content: str, metadata: dict | None = None) -> str:
        """Store a successful step sequence or action pattern."""
        return self._insert_vector(self.workflow_table, content, metadata)

    def read_workflow(self, query: str, k: int = 3) -> str:
        """Retrieve procedural patterns relevant to the current query."""
        rows = self._search_vector(self.workflow_table, query, k=k)
        if not rows:
            return "## Workflow Memory\n(no relevant patterns)\n"

        lines = [f"[sim={r['similarity']:.2f}] {r['content']}" for r in rows]
        return "## Workflow Memory\n" + "\n\n".join(lines) + "\n"

    # =========================================================================
    # 5. TOOLBOX MEMORY  (vector — semantic tool registry)
    # =========================================================================

    def write_toolbox(
        self,
        tool_name: str,
        description: str,
        parameters: dict | None = None,
        source_code: str | None = None,
        augmented: bool = False,
    ) -> str:
        """
        Register a tool definition. The description (possibly LLM-augmented)
        is embedded; name, parameters, and source are kept in metadata.
        """
        metadata = {
            "tool_name": tool_name,
            "parameters": parameters or {},
            "source_code": source_code or "",
            "augmented": augmented,
        }
        # Upsert: delete any existing entry for this tool_name first
        with self.conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.toolbox_table} WHERE metadata->>'tool_name' = %s",
                (tool_name,),
            )
        self.conn.commit()
        return self._insert_vector(self.toolbox_table, description, metadata)

    def read_toolbox(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve the top-k most relevant tools as OpenAI function-call dicts.

        Returns:
            [{"type": "function", "function": {"name": ..., "description": ...,
              "parameters": ...}}, ...]
        """
        rows = self._search_vector(self.toolbox_table, query, k=k)
        tools = []
        for r in rows:
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            params = meta.get("parameters") or {
                "type": "object", "properties": {}, "required": []
            }
            tools.append({
                "type": "function",
                "function": {
                    "name": meta.get("tool_name", "unknown"),
                    "description": r["content"],
                    "parameters": params,
                },
            })
        return tools

    # =========================================================================
    # 6. ENTITY MEMORY  (vector)
    # =========================================================================

    def write_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        *,
        llm_client=None,
        text: str | None = None,
    ) -> list[str]:
        """
        Store a named entity.

        Two call styles:
          Explicit:  write_entity("FastAPI", "library", "Python async web framework")
          Extracted: write_entity("", "", "", llm_client=client, text=query_text)
                     (mirrors the lab call; extracts entities from raw text via LLM)
        """
        if llm_client and text:
            return self._extract_and_write_entities(text, llm_client)

        if not name:
            return []

        content = f"{name}: {description}" if description else name
        metadata = {"entity_type": entity_type or "unknown", "name": name}
        return [self._insert_vector(self.entity_table, content, metadata)]

    def _extract_and_write_entities(
        self, text: str, llm_client
    ) -> list[str]:
        """
        Ask the LLM to extract entities from `text`, then store each one.
        Returns list of row ids written.
        """
        prompt = (
            "Extract all named entities from the text below.\n"
            "Return ONLY a JSON array of objects with keys: "
            '"name", "type" (person/org/system/concept/tool/other), "description".\n'
            "Return [] if there are no entities.\n\n"
            f"Text:\n{text[:2000]}"
        )
        try:
            resp = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            raw = resp.choices[0].message.content or "[]"
            # Strip markdown fences if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            entities = json.loads(raw)
        except Exception as exc:
            logger.warning("Entity extraction failed: %s", exc)
            return []

        ids = []
        for ent in entities:
            name = ent.get("name", "").strip()
            if not name:
                continue
            content = f"{name}: {ent.get('description', '')}"
            metadata = {
                "entity_type": ent.get("type", "unknown"),
                "name": name,
            }
            ids.append(self._insert_vector(self.entity_table, content, metadata))
        return ids

    def read_entity(self, query: str, k: int = 5) -> str:
        """Retrieve entities relevant to the current query."""
        rows = self._search_vector(self.entity_table, query, k=k)
        if not rows:
            return "## Entity Memory\n(no relevant entities)\n"

        lines = [f"[sim={r['similarity']:.2f}] {r['content']}" for r in rows]
        return "## Entity Memory\n" + "\n".join(lines) + "\n"

    # =========================================================================
    # 7. SUMMARY MEMORY  (vector)
    # =========================================================================

    def write_summary(
        self,
        summary_id: str,
        original_content: str,
        summary_text: str,
        description: str,
        thread_id: str | None = None,
    ) -> str:
        """
        Store a compressed summary.  The row id IS the summary_id so it can
        be used as a FK in conversational_memory.summary_id without an
        extra lookup.

        We embed the summary text (not the original) — what matters for
        retrieval is what the summary *says*, not what it was distilled from.
        """
        metadata = {
            "description": description,
            "thread_id": thread_id,
            "original_length": len(original_content),
        }
        embedding = _embed_query(summary_text, self.embedding_model)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.summary_table}
                    (id, content, embedding, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                    SET content    = EXCLUDED.content,
                        embedding  = EXCLUDED.embedding,
                        metadata   = EXCLUDED.metadata
                """,
                (summary_id, summary_text, embedding,
                 json.dumps(metadata), _now_utc()),
            )
        self.conn.commit()
        return summary_id

    def read_summary_memory(self, summary_id: str) -> str:
        """
        Exact retrieval by summary_id. Used by expand_summary().
        """
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT content FROM {self.summary_table} WHERE id = %s",
                (str(summary_id),),
            )
            row = cur.fetchone()
        return row[0] if row else f"(summary {summary_id} not found)"

    def read_summary_context(
        self,
        query: str,
        k: int = 3,
        thread_id: str | None = None,
    ) -> str:
        """
        Semantic search over summaries, optionally scoped to a thread.
        Returns short reference lines (id + description) rather than full text
        so the agent can call expand_summary(id) when it needs details.
        """
        filter_ = {"thread_id": thread_id} if thread_id else None
        rows = self._search_vector(
            self.summary_table, query, k=k, metadata_filter=filter_
        )

        # Fallback: if thread-scoped search returned nothing, try global
        if not rows and thread_id:
            rows = self._search_vector(self.summary_table, query, k=k)

        if not rows:
            return "## Summary Memory\n(no summaries)\n"

        lines = []
        for r in rows:
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            desc = meta.get("description", "summary")
            lines.append(f"[Summary ID: {r['id']}] {desc}")

        return "## Summary Memory\n" + "\n".join(lines) + "\n"

    # =========================================================================
    # 8. CODEBASE MEMORY  (vector, extension)
    # =========================================================================

    def write_codebase(
        self,
        content: str,
        repo: str,
        memory_type: str = "architecture",
        file_path: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store an architectural decision, module description, or convention."""
        meta = {**(metadata or {}), "repo": repo, "memory_type": memory_type}
        if file_path:
            meta["file_path"] = file_path
        return self._insert_vector(
            self.codebase_table, content, meta,
            extra_cols={"repo": repo, "memory_type": memory_type,
                        "file_path": file_path},
        )

    def read_codebase(
        self,
        query: str,
        repo: str | None = None,
        memory_type: str | None = None,
        k: int = 5,
    ) -> str:
        """Retrieve code-specific context relevant to the query."""
        filter_ = {}
        if repo:
            filter_["repo"] = repo
        if memory_type:
            filter_["memory_type"] = memory_type
        rows = self._search_vector(
            self.codebase_table, query, k=k,
            metadata_filter=filter_ if filter_ else None,
        )
        if not rows:
            return "## Codebase Memory\n(no relevant architecture notes)\n"

        lines = []
        for r in rows:
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            tag = meta.get("memory_type", "")
            lines.append(f"[{tag}] {r['content']}")
        return "## Codebase Memory\n" + "\n\n".join(lines) + "\n"

    # =========================================================================
    # 9. BUG MEMORY  (vector, extension)
    # =========================================================================

    def write_bug(
        self,
        content: str,
        repo: str | None = None,
        error_type: str | None = None,
        root_cause: str | None = None,
        fix_applied: str | None = None,
        file_path: str | None = None,
        resolved: bool = False,
        metadata: dict | None = None,
    ) -> str:
        """Record a bug pattern, root cause, and the fix applied."""
        meta = {**(metadata or {})}
        if repo:
            meta["repo"] = repo
        if error_type:
            meta["error_type"] = error_type
        extra = {
            "repo": repo, "error_type": error_type,
            "root_cause": root_cause, "fix_applied": fix_applied,
            "file_path": file_path, "resolved": resolved,
        }
        extra = {k: v for k, v in extra.items() if v is not None}
        return self._insert_vector(self.bug_table, content, meta, extra_cols=extra)

    def read_bug(
        self,
        query: str,
        repo: str | None = None,
        k: int = 5,
    ) -> str:
        """Find previously solved bugs that resemble the current problem."""
        filter_ = {"repo": repo} if repo else None
        rows = self._search_vector(
            self.bug_table, query, k=k, metadata_filter=filter_
        )
        if not rows:
            return "## Bug Memory\n(no similar bugs on record)\n"

        lines = [f"[sim={r['similarity']:.2f}] {r['content']}" for r in rows]
        return "## Bug Memory\n" + "\n\n".join(lines) + "\n"

    # =========================================================================
    # 10. PREFERENCE MEMORY  (vector, extension)
    # =========================================================================

    def write_preference(
        self,
        content: str,
        scope: str = "global",
        scope_value: str | None = None,
        preference_key: str | None = None,
        preference_val: str | None = None,
        confidence: float = 1.0,
        metadata: dict | None = None,
    ) -> str:
        """
        Record a user or project preference. Upserts on (key, scope, scope_value)
        if preference_key is provided.
        """
        meta = {**(metadata or {}), "scope": scope}
        if scope_value:
            meta["scope_value"] = scope_value
        if preference_key:
            meta["preference_key"] = preference_key
        if preference_val:
            meta["preference_val"] = preference_val

        # Upsert: if a canonical key exists for this scope, replace it
        if preference_key:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self.preference_table}
                    WHERE  metadata->>'preference_key' = %s
                      AND  metadata->>'scope' = %s
                      AND  (%s::text IS NULL
                            OR metadata->>'scope_value' = %s)
                    """,
                    (preference_key, scope, scope_value, scope_value),
                )
            self.conn.commit()

        extra = {
            "scope": scope,
            "scope_value": scope_value,
            "preference_key": preference_key,
            "preference_val": preference_val,
            "confidence": confidence,
        }
        extra = {k: v for k, v in extra.items() if v is not None}
        return self._insert_vector(
            self.preference_table, content, meta, extra_cols=extra
        )

    def read_preference(
        self,
        query: str,
        scope: str | None = None,
        scope_value: str | None = None,
        k: int = 10,
    ) -> str:
        """Retrieve relevant preferences (used at session start)."""
        filter_ = {}
        if scope:
            filter_["scope"] = scope
        if scope_value:
            filter_["scope_value"] = scope_value
        rows = self._search_vector(
            self.preference_table, query, k=k,
            metadata_filter=filter_ if filter_ else None,
        )
        if not rows:
            return "## Preference Memory\n(no preferences on record)\n"

        lines = [r["content"] for r in rows]
        return "## Preference Memory\n" + "\n".join(lines) + "\n"

    # =========================================================================
    # Context builder  (full partitioned prompt used by get_context MCP tool)
    # =========================================================================

    def build_context(
        self,
        query: str,
        thread_id: str,
        repo: str | None = None,
    ) -> str:
        """
        Assemble a partitioned context window from all memory stores.
        Mirrors the context-building logic in L5 call_agent().
        """
        parts = [
            f"# Question\n{query}\n",
            self.read_conversational_memory(thread_id),
            self.read_knowledge_base(query),
            self.read_workflow(query),
            self.read_entity(query),
            self.read_summary_context(query, thread_id=thread_id),
        ]
        if repo:
            parts.append(self.read_codebase(query, repo=repo))
            parts.append(self.read_bug(query, repo=repo))

        parts.append(self.read_preference("coding preferences", scope="global"))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Toolbox  (mirrors helper.py Toolbox; used to register callable tools)
# ---------------------------------------------------------------------------

@dataclass
class _ToolMeta:
    name: str
    description: str
    parameters: dict
    source_code: str = ""
    augmented: bool = False
    fn: Callable = field(repr=False, default=None)


class Toolbox:
    """
    Decorator-based tool registry that stores tool descriptions as vectors
    in toolbox_memory for semantic retrieval.

    Usage:
        toolbox = Toolbox(memory_manager, llm_client, embedding_model)

        @toolbox.register_tool(augment=True)
        def search_web(query: str) -> str:
            "Search the web for current information."
            ...
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        llm_client,
        embedding_model,
    ) -> None:
        self.mm = memory_manager
        self.llm = llm_client
        self.embedding_model = embedding_model
        self._registry: dict[str, _ToolMeta] = {}

    # -- Registration ---------------------------------------------------------

    def register_tool(self, augment: bool = False):
        """Decorator factory. Embeds and stores the tool in toolbox_memory."""

        def decorator(fn: Callable) -> Callable:
            name = fn.__name__
            raw_doc = inspect.getdoc(fn) or ""
            params = self._extract_parameters(fn)

            description = (
                self._augment_docstring(fn, raw_doc) if augment else raw_doc
            )

            meta = _ToolMeta(
                name=name,
                description=description,
                parameters=params,
                source_code=inspect.getsource(fn),
                augmented=augment,
                fn=fn,
            )
            self._registry[name] = meta

            try:
                self.mm.write_toolbox(
                    tool_name=name,
                    description=description,
                    parameters=params,
                    source_code=meta.source_code,
                    augmented=augment,
                )
                logger.info("Registered tool: %s (augmented=%s)", name, augment)
            except Exception as exc:
                logger.warning("Failed to persist tool %s: %s", name, exc)

            return fn

        return decorator

    def call(self, tool_name: str, **kwargs) -> Any:
        """Invoke a registered tool by name."""
        meta = self._registry.get(tool_name)
        if meta is None:
            raise KeyError(f"Tool '{tool_name}' is not registered in this Toolbox")
        return meta.fn(**kwargs)

    # -- Internal helpers -----------------------------------------------------

    def _extract_parameters(self, fn: Callable) -> dict:
        """Build an OpenAI-compatible JSON schema from function signature."""
        sig = inspect.signature(fn)
        type_map = {
            "str": "string", "int": "integer", "float": "number",
            "bool": "boolean", "list": "array", "dict": "object",
        }
        props, required = {}, []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                ptype = "string"
            else:
                ann_name = getattr(ann, "__name__", str(ann))
                ptype = type_map.get(ann_name, "string")

            props[pname] = {"type": ptype}
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        return {"type": "object", "properties": props, "required": required}

    def _augment_docstring(self, fn: Callable, raw_doc: str) -> str:
        """
        Use the LLM to generate a richer description from the docstring
        + source code. Falls back to raw_doc on any error.
        """
        try:
            source = inspect.getsource(fn)
            prompt = (
                "You are improving a Python tool's description for semantic "
                "retrieval in a vector database.\n\n"
                "Given the docstring and source code, write a comprehensive "
                "description (3-5 sentences) that covers:\n"
                "- What the tool does\n"
                "- When to use it\n"
                "- What inputs it expects\n"
                "- What it returns\n\n"
                "Return ONLY the description text, no headings or code.\n\n"
                f"Docstring:\n{raw_doc}\n\n"
                f"Source:\n{source[:2000]}"
            )
            resp = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            augmented = (resp.choices[0].message.content or "").strip()
            return augmented if augmented else raw_doc
        except Exception as exc:
            logger.warning("Docstring augmentation failed for %s: %s", fn.__name__, exc)
            return raw_doc
