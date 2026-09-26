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

## Metadata

An `IdeaNote` may contain YAML front matter.

The metadata belongs to the `IdeaNote` and is stored separately from its
Markdown body.

Metadata is preserved when the note is serialized.

Metadata does not determine whether a file is an `IdeaNote`; the file
extension does.