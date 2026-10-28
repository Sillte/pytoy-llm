# Tool Design Policy

## Purpose

`tool` owns contracts and runtime support shared by LLM-facing tools and
extension authors. Its public API is intentionally distinct from the concrete
capabilities in `tools`.

## Responsibilities

- `ToolError` and `ToolErrorKind` define expected tool failures.
- `plugin` owns the extension-facing tool contract, dynamic loaders, and
  registry.
- `tools` implementations may depend on `tool`; `tool` must not depend on
  concrete implementations in `tools`.
- Keep internal implementation modules behind the package APIs. Import errors
  from `pytoy_llm.tool`; import plugin extensions from `pytoy_llm.tool.plugin`.
