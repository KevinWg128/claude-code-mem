-- schema.sql — Agent Memory MCP database schema
-- Requires PostgreSQL 14+ with pgvector extension
-- Safe to re-run (all statements use IF NOT EXISTS)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- SQL tables (exact retrieval, no embeddings)
-- ============================================================================

CREATE TABLE IF NOT EXISTS conversational_memory (
    id          UUID PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_conv_thread
    ON conversational_memory (thread_id, created_at);

CREATE TABLE IF NOT EXISTS tool_log_memory (
    id            UUID PRIMARY KEY,
    thread_id     TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    input_args    TEXT DEFAULT '',
    output        TEXT DEFAULT '',
    status        TEXT DEFAULT 'success',
    error_message TEXT,
    duration_ms   FLOAT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata      JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_tool_log_thread
    ON tool_log_memory (thread_id, created_at);

-- ============================================================================
-- Vector tables (semantic search via pgvector)
-- Default embedding dimension: 768 (sentence-transformers/paraphrase-mpnet-base-v2)
-- Change to 1536 for openai/text-embedding-3-small
-- Change to 3072 for openai/text-embedding-3-large
-- ============================================================================

CREATE TABLE IF NOT EXISTS semantic_memory (
    id          UUID PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(768),
    metadata    JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workflow_memory (
    id          UUID PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(768),
    metadata    JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS toolbox_memory (
    id          UUID PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(768),
    metadata    JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS entity_memory (
    id          UUID PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(768),
    metadata    JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS summary_memory (
    id          UUID PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(768),
    metadata    JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS codebase_memory (
    id           UUID PRIMARY KEY,
    content      TEXT NOT NULL,
    embedding    vector(768),
    metadata     JSONB DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    repo         TEXT,
    memory_type  TEXT,
    file_path    TEXT
);

CREATE TABLE IF NOT EXISTS bug_memory (
    id           UUID PRIMARY KEY,
    content      TEXT NOT NULL,
    embedding    vector(768),
    metadata     JSONB DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    repo         TEXT,
    error_type   TEXT,
    root_cause   TEXT,
    fix_applied  TEXT,
    file_path    TEXT,
    resolved     BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS preference_memory (
    id              UUID PRIMARY KEY,
    content         TEXT NOT NULL,
    embedding       vector(768),
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    scope           TEXT DEFAULT 'global',
    scope_value     TEXT,
    preference_key  TEXT,
    preference_val  TEXT,
    confidence      FLOAT DEFAULT 1.0
);

-- ============================================================================
-- Vector indexes (IVFFlat — good default for < 1M rows per table)
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_semantic_embedding
    ON semantic_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_workflow_embedding
    ON workflow_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_toolbox_embedding
    ON toolbox_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_entity_embedding
    ON entity_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_summary_embedding
    ON summary_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_codebase_embedding
    ON codebase_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_bug_embedding
    ON bug_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_preference_embedding
    ON preference_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
