# Task Audit Design Policy

## Purpose

`task.audit` turns a completed task result into an audit snapshot and provides
ways to export or present that snapshot. It records completed work; it does not
run or manage task execution.

## Responsibility Boundaries

- `TaskAuditLog` and its record types define the audit snapshot, including task
  start and end records, invocation records, and links between them.
- `TaskAuditor` is the package's public entry point. It creates the snapshot
  from a `TaskResult` and exposes its supported output forms.
- `dump` exports the complete audit snapshot as JSON.
- `make_llm_messages_audit_section` returns a supplementary material section
  containing only the final LLM message history and token totals. It is not a
  replacement for the complete audit export.
- The caller composes the section, choosing its Markdown header depth and where
  it belongs among other prompt sections.

## Design Constraints

- Keep execution lifecycle, retries, and task state management outside
  `task.audit`.
- Keep audit record types internal unless they are deliberately added to the
  package's public exports. External consumers should import `TaskAuditor`
  from `pytoy_llm.task.audit`.
