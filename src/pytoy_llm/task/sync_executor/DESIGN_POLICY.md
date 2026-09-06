# Sync Executor Design Policy

## Purpose

`TaskSyncExecutor` provides a small execution path for trying a task
synchronously. It is intended for callers that need the task result directly
and do not need execution tracking or background processing.

## Design

- `execute` runs the task in the calling thread.
- The request input and the request activity sink are passed to the task
  specification unchanged.
- When a request has no context state, a new `TaskContextState` is created for
  that execution.
- The task outcome is wrapped in a `TaskExit` with the request ID.
- The method returns only after the task specification has returned.

The executor deliberately does not create a thread, register an execution, or
provide lifecycle events. Those responsibilities belong to the asynchronous
task execution mechanism.

## Error and Stability Boundaries

This executor is a synchronous trial mechanism, not a reliability boundary.
Exceptions raised by the task specification are allowed to propagate to the
caller. In particular, it does not provide the exception-to-`Error`
translation performed by the asynchronous execution path.

The executor should remain simple even if that means it has fewer protections
than the asynchronous path. Its contract is to make a synchronous attempt and
return the task outcome when the attempt completes; it does not guarantee
retrying, cancellation, timeout handling, isolation, or recovery from task
failures.

## Usage Boundary

Use `TaskSyncExecutor` when:

- the caller can block while the task runs;
- immediate access to the resulting `TaskExit` is useful; and
- the caller can handle exceptions from the task directly.

Use the asynchronous task execution mechanism when the caller needs background
execution, execution status, lifecycle notifications, or centralized failure
handling.

Changes to this package should preserve the synchronous, calling-thread
behavior unless the executor is intentionally replaced by a different
abstraction.