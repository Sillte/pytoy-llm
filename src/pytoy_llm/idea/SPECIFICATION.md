# IdeaSpace Specification

## IdeaSpace

An `IdeaSpace` represents a directory within an IdeaSpace root.

The root is identified by the `.idea_space_root` marker file, or explicitly
specified for an `IdeaSpace`.

Every directory under the root is an `IdeaSpace`, unless it is a reserved
internal directory.

All paths are represented relative to the IdeaSpace root.

An `IdeaSpace` does not require its own marker file.

An `IdeaSpace` must not be named `.space_meta`.

## IdeaNote

An `IdeaNote` represents a Markdown file (`.md`) within an `IdeaSpace`.

Only Markdown files are treated as `IdeaNote`s.

Other files are not `IdeaNote`s and may be accessed through separate
artifact APIs.

For operations that accept either an `IdeaSpace` path or an `IdeaNote` path,
the path kind is determined by its final path component: a component ending in
`.md` identifies an `IdeaNote`; any other component identifies an `IdeaSpace`.
This classification is based on the path suffix, not the filesystem entry type.

## Metadata

An `IdeaNote` may contain YAML front matter.

The metadata belongs to the `IdeaNote` and is stored separately from its
Markdown body.

Metadata is preserved when the note is serialized.

Metadata does not determine whether a file is an `IdeaNote`; the file
extension does.


## Convention

A `Convention` is an `IdeaNote` that defines local conventions for
organizing, interpreting, creating, or maintaining `IdeaNotes` within
an `IdeaSpace`.

The conventional filename for a `Convention` is `.convention.md`.

A Convention defined in an `IdeaSpace` applies to `IdeaNotes` directly
contained in that `IdeaSpace`. It is inherited by descendant
`IdeaSpace`s and applies to the `IdeaNotes` directly contained in each
of those `IdeaSpace`s.

When multiple conventions apply to an `IdeaNote`, a convention defined
in a deeper `IdeaSpace` takes precedence over conventions defined in
shallower `IdeaSpace`s.


A Convention does not explicitly store its scope. Its scope is
determined by the hierarchy of the `IdeaSpace` containing it.

Convention `IdeaNotes` may be created, read, updated, and deleted by
both LLMs and humans through the same CRUD operations as other
`IdeaNotes`.

