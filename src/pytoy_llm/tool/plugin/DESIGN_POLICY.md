# Plugin Design Policy

## Purpose

`plugin` loads developer-authored Python modules that provide callable tools to
LLM agents. `LLMToolsPlugin` is the single extension-facing contract. The
`LLMPluginCollection` in `models.py` is an internal runtime aggregate and is
not exported from the package root.
## Contract and Trust

- Concrete `LLMToolsPlugin` subclasses have no-argument constructors and expose
  one or more callables through the `tools` property. They may override
  `guidance()` to describe when and how to use the tools; its default is `None`.
  Guidance is plain text intended for possible inclusion in a system prompt. It
  is not automatically inserted or formatted by the plugin package.
- Plugin modules are selected by application/developer code, never by an LLM
  request.
- Loading executes arbitrary Python in the host process with its permissions.
  This package is not a sandbox; only trusted code may be loaded.

## Responsibilities

- `FileLoader` imports one standalone Python file and collects concrete
  `LLMToolsPlugin` subclasses defined in it. Each file must define at least one
  concrete plugin class. Use `DirectoryLoader` for plugins with sibling modules
  and relative imports.
- `DirectoryLoader` treats the selected directory as a synthetic package and
  examines only direct `.py` files, in filename order. Module stems must be
  valid Python identifiers. It skips `__init__.py` and modules without a
  concrete plugin class, but requires at least one plugin overall. Relative
  imports between direct modules use Python's package-relative syntax, such as
  `from .helpers import helper`.
- The supported layout is flat. Nested directories and subpackages are not
  traversed, discovered, or included in the content hash. Do not place plugin
  modules or their helper modules in nested directories or rely on imports from
  them; flatten such files into the selected directory.
- File modules are cached by resolved path and source hash. Directory package
  namespaces are cached by resolved directory and the content manifest of its
  direct `.py` files. Unchanged inputs reuse modules, but each load creates new
  plugin instances and a new collection. Mutable module-level state remains
  shared within a cached module. A changed version replaces the prior namespace
  only after successful loading. Identical content at different paths is not
  shared. Errors while executing modules still propagate.
- `LLMPluginCollection.from_plugins` flattens the tools from loaded plugin
  instances and snapshots their guidance. Its `guidances` property returns
  non-empty plugin guidance strings in load order.
- `LLMToolPluginRegistry.from_any` loads one file or directory and exposes the
  collected callables and guidance. It implements the existing `LLMToolset`
  shape through its `tools` property.
- Tool callables must have a non-empty `__name__` other than `<lambda>`, and
  names must be unique within the loaded collection.
- System-prompt formatting remains the responsibility of the composer; plugin
  guidance is plain text and does not depend on composer types.
- Plugin tools remain ordinary callables and follow the existing tool
  registration and error-handling contracts.
