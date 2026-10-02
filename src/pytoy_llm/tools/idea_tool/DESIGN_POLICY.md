# IdeaTool Design Policy

## Purpose

`IdeaTool` exposes an LLM-facing toolset for discovering, inspecting, and
mutating one or more configured `IdeaSpace` roots. It adapts the `idea` package
to tool-call inputs, outputs, and expected errors; it does not own note parsing,
space traversal, or link-resolution algorithms.

## Public API

Import `IdeaTool` from `pytoy_llm.tools.idea_tool`. The class is the package's
public facade. Its `discovery`, `inspection`, and `mutation` properties group
the operations, and `tools` provides the callable operations for registration
with an LLM agent.

The implementation classes, boundary decorators, semantic types, and Pydantic
models in this package's other modules are internal. Consumers should not
depend on those modules directly. Keep the package root's exports intentional.

## Responsibilities

- `IdeaTool` configures namespaces and local URI routes, validates that each
  route points to its corresponding `IdeaSpace` root, and composes the
  operation groups.
- Discovery lists IdeaSpace and IdeaNote paths and convention locations; it
  does not return note contents.
- Inspection reads note, metadata, convention, link, and working-context data.
- Mutation creates, updates, moves, and deletes IdeaSpaces and IdeaNotes.
- Tool-facing models and semantic types describe the operation schemas and
  results; they do not replace the `idea` package's domain types.

Keep the dependency from `idea_tool` toward the `idea` package's public API and
the shared `tools.errors` contract. The `idea` package must not depend on
`idea_tool`, and `IdeaTool` must not depend on `WorkspaceExplorer` internals.

## Namespaces and Paths

Each namespace identifies one configured IdeaSpace root. When an operation's
namespace is omitted, it targets the configured default namespace. All
IdeaSpace and IdeaNote paths are relative to that selected root; resolution and
root containment remain the responsibility of the `idea` package. A local URI
route for the `idea` scheme must match the configured root for its namespace.

Operations on a note or space stay within one namespace. Cross-namespace links
may be represented and inspected, but a move does not move content between
roots or rewrite links.

## Mutation Guarantees

- Never allow a mutation to escape its selected IdeaSpace root or alter the
  reserved tool-metadata area through note or space paths.
- Do not overwrite an existing destination during a move. A move within the
  same namespace leaves links unchanged; moving a convention note or subspace
  may change the conventions effective at its new location.
- Delete only empty IdeaSpace directories. Deleting an IdeaNote is permanent.
- `write_idea_note` replaces the existing note body and metadata; metadata
  updates otherwise preserve the body unless the operation explicitly says
  otherwise.
- Expected input, permission, missing-path, and I/O failures return
  `ToolError`. Do not turn unexpected exceptions into ordinary tool results.

Keep these guarantees enforced by the owning mutation operation and `IdeaSpace`
path resolution rather than duplicating path policy in each caller.

## Working Context

The lifecycle hooks `mark_llm_start` and `mark_llm_finished` record the latest
invocation timestamps in each configured IdeaSpace's reserved tool-metadata
area. This is the tool's intentional metadata side effect, separate from
IdeaNote content mutations.

## Testing and Evolution

Test behavior through the package-root `IdeaTool` API. Cover namespace
selection, root containment, collision handling, and the persistence behavior
of mutations when changing those contracts. Add public exports only when a
consumer needs a stable concept beyond the facade and its tool operations.