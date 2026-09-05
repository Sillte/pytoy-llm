# AGENTS.md

This repository is a personal project for exploring Python-based
editor plugin development and software architecture.

Follow the applicable `DESIGN_POLICY.md` files when modifying the
codebase.

## Documentation

- `README.md`: project overview and user-facing documentation.
- `DESIGN_POLICY.md`: design principles and architectural decisions.
- `AGENTS.md`: instructions for AI agents working in this repository.

A `DESIGN_POLICY.md` applies to its directory and descendants.
More specific policies refine broader policies.

Avoid documenting information that can be reliably recovered from the code or other sources of truth.

## Architecture

- Preserve dependency direction and responsibility boundaries.
- Prefer simple designs over unnecessary abstractions.
- Keep implementation details behind public APIs.
- Avoid unnecessary global state and import-time side effects.

## Testing

- Test public behavior, not implementation details.
- Prefer simple, behavior-oriented tests.
- Avoid unnecessary mocks.
- Use `threading.Event` for asynchronous execution instead of `sleep`.
- Inspect the near tests before implementing tests.
- Do not add tests merely to increase coverage.
- Do not accept multiple exception types unless the public contract explicitly
  allows those alternatives.
