# Tools Design Policy

## Purpose

The `tools` package provides concrete LLM-facing capabilities, including
workspace exploration and IdeaSpace operations. These capabilities have
separate access and mutation policies.

## Responsibility Boundaries

- `WorkspaceExplorer` is the public facade for read-only workspace exploration;
  its specific policy is in `workspace_explorer/DESIGN_POLICY.md`.
- `IdeaTool` is the public facade for discovery, inspection, and mutation of
  configured IdeaSpaces; its specific policy is in `idea_tool/DESIGN_POLICY.md`.
- Shared tool contracts, errors, and plugin loading belong to `pytoy_llm.tool`.
  Concrete implementations may depend on that package, not the reverse.
- Discovery, inspection, and search responsibilities remain separate within
  each tool.
- Internal implementation modules should not be required by package users.

## Errors

Expected tool failures are returned as `ToolError` values from
`pytoy_llm.tool`. Implementations should distinguish invalid arguments, missing
paths, permission failures, and resource limits where the cause is known.
Unexpected exceptions should not be used as the normal tool contract.
