# Idea Design Policy

## Purpose

`idea` models a workspace of Markdown notes and the links between them. It
provides note parsing, space traversal, link resolution, and graph queries for
editor integrations.

## Public API

The package root is the public API boundary. Consumers should import public
concepts from `pytoy_llm.idea`, including:

- `IdeaNote`, `IdeaSpace`, and `IdeaGraph`
- `IdeaLink` and `UnresolvedIdeaLink`
- `TextPosition`, `TextRange`, `LineLocation`, `AnchorLocation`, and `Location`
- `LinkReachabilityChecker`

Parser, resolver, converter, infrastructure, and URI helper modules are
implementation details and should not be imported by ordinary consumers.

Link resolution implementations belong under `link_resolvers`. `IdeaGraph`
coordinates graph operations but does not own Markdown, WikiLink, or URI
resolution algorithms.

## Responsibilities

- `IdeaNote` represents one note, including YAML front matter and Markdown body.
- `IdeaSpace` represents a rooted note directory and traverses notes and
  subspaces.
- `IdeaGraph` resolves links and provides graph queries such as backlinks.
- `IdeaLink` represents a resolved link exposed to consumers.
- `LinkReachabilityChecker` performs reachability checks. File-system and HTTP
  I/O must remain outside `IdeaLink`.

## Link Locations

`LineLocation` represents a single line and column. `AnchorLocation` represents
an anchor. A link fragment such as `#L10-L20` currently retains only its start
line; range targets require a separate location type and should not extend
`LineLocation` with range fields.

## Dependency Direction

Domain models must not depend on concrete link resolvers, extractors, or
infrastructure implementations. Resolvers and adapters may depend on domain
contracts and models. URI conversion and reachability mechanisms remain
internal implementation details.

## Evolution

Add behavior-oriented tests against the package-root public API before
expanding the public surface. Keep experimental parser behavior explicitly
scoped rather than exposing parser implementation types.
