# Task Execution Design Policy

## Purpose

`task.execution` owns the lifecycle and lookup of one submitted task. Session
history and shared context belong to `task.session`.

## Boundaries

- `TaskExecutionHandler` is the public entry point for creating, querying,
  starting, and observing executions.
- `TaskExecution` is an internal lifecycle object. The handler exposes its
  identifier, state, and events without exposing the object itself.
- `TaskExecutionManager` is the internal registry used by handlers; it is not
  the client-facing API.
- `TaskSession` coordinates executions through the handler and must not
  duplicate execution state or implementation.

## Public Contract

Types appearing in the handler's arguments, return values, properties, events,
or query contract are public and must be exported from
`pytoy_llm.task.execution`:

- `TaskExecutionID`
- `TaskExecutionStatus`
- `TaskExecutionExit`
- `TaskExecutionHooks`
- `TaskExecutionQuery`

Other implementation types remain internal. External code should import public
symbols from the package, not from `handler.py`, `models.py`, or `manager.py`.

`TaskExecutionQuery` uses `TaskExecutionStatus` as its status vocabulary and
returns handlers rather than internal execution objects.

## Constraints

- Keep implementation details behind the package API.
- Review exported contract types when changing a handler signature or event
  payload.
- Keep cancellation separate from execution completion unless cancellation is
  explicitly implemented.
