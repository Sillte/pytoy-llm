# Tools Design Policy

## Purpose

The `tools` package provides read-only tools for LLM agents to inspect and
search workspace contents.

## Responsibility Boundaries

- `WorkspaceExplorer` is the public facade for workspace exploration.
- Discovery, inspection, and search responsibilities remain separate.
- `WorkspaceAccess` owns workspace path validation and access policy.
- Internal implementation modules should not be required by package users.

## Safety

- All paths are interpreted relative to the workspace root.
- Absolute paths, parent traversal, and symlink or junction paths resolving
  outside the workspace must be rejected.
- Workspace tools must not create, modify, or delete files.
- Path validation must remain centralized in `WorkspaceAccess`.

## Resource Limits

File reads and searches must remain bounded by explicit resource limits, such
as file size and result count. Exceeding a limit returns
`ToolErrorKind.RESOURCE_LIMIT` rather than loading unbounded content.

## Errors

Expected tool failures are returned as `ToolError` values. Implementations
should distinguish invalid arguments, missing paths, permission failures, and
resource limits where the cause is known. Unexpected exceptions should not be
used as the normal tool contract.

## Exclusions

Default exclusions exist to avoid generated files and dependency trees during
workspace exploration. An explicitly supplied empty exclusion collection means
that no configurable exclusions were requested. Any unconditional exclusions
must be enforced and documented by the path-gathering policy.
