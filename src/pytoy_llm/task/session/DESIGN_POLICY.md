# Task Session Design Policy

## Purpose

`TaskSession` groups multiple task submissions that share context and belong to
one logical interaction or workflow.

## Responsibilities

- `TaskExecution` owns the lifecycle of one task execution.
- `TaskSessionHandler` is the public API for creating, submitting to, and
  observing a session.
- `TaskSession` owns the shared `TaskContextState` and task result history.
- `TaskSessionRequest` is the input for creating a session. Its default `kind`
  is `DEFAULT_KIND`; a query with no kind does not filter by kind.
- `TaskRecord` is the session's task-level history record. It retains the
  request, status, output or exception, and activities, but not the full exit
  payload or task result.
- A successful execution updates the session context for later executions.
- `TaskExecutionManager` manages execution lookup and lifecycle access; it is
  not the session history store.
- `TaskSessionManager` manages session lookup and selection.

A session should coordinate executions through `TaskExecutionHandler` without
exposing or reimplementing the internal `TaskExecution` object.

## Lifecycle

Submitting a task is accepted only while the session is `idle`, and changes the
session to `running`. Every task exit, including failure, returns the session
to `idle` and records the task result or exception. A completed session is
entered explicitly and cannot accept further tasks.

A request with an explicit context state keeps that state for the execution.
Otherwise, the session's current context state is used.

Task completion and session completion are separate events. `on_task_exit`
reports an individual task exit, while `on_session_exit` reports explicit
session completion.

`terminate` is a strong destruction operation. It removes the session from its
manager, disposes subscriptions and event resources, clears retained records,
and makes the session inaccessible through its handler. It does not imply that
an already-running task can be cancelled; task cancellation is a separate
execution concern.
