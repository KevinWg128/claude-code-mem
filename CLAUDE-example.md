# CLAUDE.md — Agent Memory Protocol (MANDATORY)

## ⚠️ CRITICAL: Memory MCP Tools Are REQUIRED, Not Optional

You have access to an `agent-memory` MCP server. **Tool logging is handled automatically by hooks**, but you MUST still use the semantic memory tools below. Failure to do so means losing critical context across sessions.

## MANDATORY Actions (Non-Negotiable)

### At Session Start (FIRST THING, before any other work):
```
Call: get_context(query="<user's task description>", repo="claude-code-mem-mcp")
```
Do this BEFORE reading files, running commands, or responding. No exceptions.

### After Every Bug Fix:
```
Call: remember_bug(
  content="<what broke, why, and how you fixed it>",
  repo="claude-code-mem-mcp",
  error_type="<exception class>",
  root_cause="<one-line cause>",
  fix_applied="<one-line fix>"
)
```

### After Learning Architecture/Convention:
```
Call: remember_codebase(
  content="<what you learned about the codebase>",
  repo="claude-code-mem-mcp",
  memory_type="architecture|module|convention|dependency|constraint"
)
```

### After Observing User Preferences:
```
Call: remember_preference(
  content="<the preference>",
  preference_key="<short key>",
  preference_val="<value>"
)
```

### At Session End (LAST THING):
1. Get the transcript:
```
Call: summarise_thread(thread_id="claude-code")
```
2. Read the returned transcript, then generate a summary with headings:
   `Work Done`, `Decisions Made`, `Open Items`, `Entities & References`
3. Store it:
```
Call: summarise_thread(
  thread_id="claude-code",
  summary_text="<your generated summary>",
  description="<8-12 word label>"
)
```

## Why This Matters
Without these calls, every session starts from zero. The hooks handle tool-call logging automatically, but only YOU can decide what architectural knowledge, bugs, and preferences to persist. This is your long-term memory — use it.

