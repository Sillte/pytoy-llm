# Task Session Design Policy

## Purpose

`TaskSession` groups multiple task executions that share context and belong to
one logical interaction or workflow.

## Responsibilities

- `TaskExecution` owns the lifecycle of one task execution.
- `TaskSession` owns the shared `TaskContextState` and task result history.
- A successful execution updates the session context for later executions.
- `TaskExecutionManager` manages execution lookup and lifecycle access; it is not
  the session history store.
- `TaskSessionManager` manages session lookup and selection.

A session should coordinate executions without reimplementing task execution,
threading, or invocation behavior.

## Lifecycle

Submitting a task is accepted only while the session is `idle`, and changes the
session to `running`. Every task exit, including failure, returns the session
to `idle` and records the task result or exception. A completed session is
entered explicitly and cannot accept further tasks.

A request with an explicit context state keeps that state for the execution.
Otherwise, the session's current context state is used.
