# Plugin Design Policy

## Purpose

`plugin` loads developer-authored Python modules that provide callable tools to
LLM agents. `LLMPluginTool` is the extension-facing ABC for one tool;
`LLMPluginTools` is the loader-created collection of those implementations.

## Contract and Trust

- Concrete `LLMPluginTool` subclasses have no-argument constructors and expose
  their callable through the `tool` property.
- Plugin modules are selected by application/developer code, never by an LLM
  request.
- Loading executes arbitrary Python in the host process with its permissions.
  This package is not a sandbox; only trusted code may be loaded.

## Responsibilities

- `FileLoader` imports one Python file and collects concrete tool subclasses
  defined in that module. Each loaded file must define at least one concrete
  tool class.
- `DirectoryLoader` loads only `.py` files directly inside the selected
  directory, in filename order; it does not recurse and skips `__init__.py`.
  Every other `.py` file in the directory must define a concrete tool class.
- `LLMPluginTools` snapshots the tool instances and callables when constructed.
  The registry can load a file or directory and exposes all collected callables
  through its `tools` property as an `LLMToolSet`.
- Registered tool callables must have a non-empty `__name__` other than
  `<lambda>`, and tool names must be unique across the registry.
- Plugin tools remain ordinary callables and follow the existing tool
  registration and error-handling contracts.
