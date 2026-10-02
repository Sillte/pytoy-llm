# Tools Design Policy

## Purpose

The `tools` package provides LLM-facing tools for workspace exploration and
IdeaSpace operations. These are separate capabilities with different access
and mutation policies.

## Responsibility Boundaries

- `WorkspaceExplorer` is the public facade for read-only workspace exploration;
  its specific policy is in `workspace_explorer/DESIGN_POLICY.md`.
- `IdeaTool` is the public facade for discovery, inspection, and mutation of
  configured IdeaSpaces; its specific policy is in `idea_tool/DESIGN_POLICY.md`.
- Discovery, inspection, and search responsibilities remain separate within
  each tool.
- Internal implementation modules should not be required by package users.

## Errors

Expected tool failures are returned as `ToolError` values. Implementations
should distinguish invalid arguments, missing paths, permission failures, and
resource limits where the cause is known. Unexpected exceptions should not be
used as the normal tool contract.
