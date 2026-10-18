# Task Execution Design Policy

## Purpose

`task.execution` owns the lifecycle and lookup of one submitted task. Session
history and shared context belong to `task.session`.

## Boundaries

- `TaskExecutionHandler` is the public entry point for creating, querying,
  starting, canceling, and observing executions.
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

Cancellation can be requested before or during an execution. `on_cancel` reports
that the request was accepted. Canceling a created execution transitions it
directly to `canceled`; it has not started, so it does not emit `on_exit`. A
running execution transitions to `canceling`; its `CancelToken` is shared with
the task and observed at invocation boundaries, so an invocation already in
progress is not interrupted. Once a started execution terminates, it emits
`on_exit` exactly once, including after cancellation; a canceled exit carries a
`TaskCanceledException` outcome and transitions to `canceled`. The execution
manager deregisters an unstarted execution on cancellation and a started
execution on exit.

## Constraints

- Keep implementation details behind the package API.
- Review exported contract types when changing a handler signature or event
  payload.
- Keep cancellation separate from execution completion unless cancellation is
  explicitly implemented.
