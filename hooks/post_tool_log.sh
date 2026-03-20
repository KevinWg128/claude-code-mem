#!/bin/bash
# post_tool_log.sh — Automatic hook that logs tool calls to agent-memory MCP
#
# Reads the hook event JSON from stdin, extracts tool_name + result,
# and writes to the tool_log_memory table via a direct psycopg2 insert.
#
# This runs AUTOMATICALLY after Claude Code finishes a tool call.
# The LLM cannot skip this — it is enforced by the hook system.

set -euo pipefail

DB_DSN="${MEMORY_DB_DSN:-postgresql://agent_mcp:agent_mcp_password@localhost:5432/agent_memory}"
THREAD_ID="${MEMORY_DEFAULT_THREAD:-claude-code}"
PYTHON="${HOOK_PYTHON:-/Users/foodsup/side-projects/claude-code-mem-mcp/.venv/bin/python}"

# Read the hook event JSON from stdin
EVENT_JSON=$(cat)

# Extract fields from the hook event
TOOL_NAME=$(echo "$EVENT_JSON" | "$PYTHON" -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('tool_name', 'unknown'))
" 2>/dev/null || echo "unknown")

TOOL_INPUT=$(echo "$EVENT_JSON" | "$PYTHON" -c "
import sys, json
data = json.load(sys.stdin)
inp = data.get('tool_input', {})
# Truncate to 500 chars to keep logs lean
s = json.dumps(inp)[:500]
print(s)
" 2>/dev/null || echo "{}")

TOOL_OUTPUT=$(echo "$EVENT_JSON" | "$PYTHON" -c "
import sys, json
data = json.load(sys.stdin)
out = data.get('tool_output', '')
if isinstance(out, dict):
    out = json.dumps(out)
# Truncate to 1000 chars
print(str(out)[:1000])
" 2>/dev/null || echo "")

# Write directly to the database (fast, no model loading)
"$PYTHON" -c "
import psycopg2, json, uuid, sys
from datetime import datetime, timezone

dsn = '$DB_DSN'
try:
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute('''
            INSERT INTO tool_log_memory 
                (id, thread_id, tool_name, input_args, output, status, created_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            str(uuid.uuid4()),
            '$THREAD_ID',
            sys.argv[1],
            sys.argv[2],
            sys.argv[3],
            'success',
            datetime.now(timezone.utc),
            json.dumps({'source': 'hook', 'auto_logged': True}),
        ))
    conn.close()
except Exception as e:
    # Silent failure — hooks should never block Claude Code
    sys.stderr.write(f'Hook log failed: {e}\n')
" "$TOOL_NAME" "$TOOL_INPUT" "$TOOL_OUTPUT" 2>/dev/null || true
