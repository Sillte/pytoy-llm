# JSON Codecs Design Policy

## Purpose

`json_codecs` translates between pytoy-llm message models and external JSON
wire formats. `CompletionMessagesCodec` currently targets message objects and
message arrays from the OpenAI Chat Completions API. Completion response
metadata and Responses API objects are outside this package's current scope.

## Public API

The package root is the public API boundary. Consumers should import codecs
from `pytoy_llm.json_codecs`. Codec implementation modules are internal
implementation details.

## Responsibilities

- Preserve supported text messages, content-part arrays, assistant tool calls,
  and tool results when converting between `LLMMessage` and Completion JSON.
- Combine assistant text and multiple `ToolCallRequestPart` instances into a
  single assistant message with a `tool_calls` array when encoding.
- Preserve unsupported or malformed message structures as `OpaquePart` rather
  than silently discarding their data when decoding.
- Keep provider-specific client adapters delegated to the appropriate codec;
  wire-format conversion should not be duplicated in client adapters.

## Unsupported Structures

`OpaquePart` is the fallback for Completion message structures that the domain
models do not represent yet, such as unknown roles or malformed tool calls.
When extending the supported format, add a domain Part type and codec behavior
only when the structure has a stable meaning for pytoy-llm. Otherwise, retain
it as an opaque value.

## Known Information Loss

The current decoder can silently discard unsupported fields when a message has
an otherwise supported role and content. For example, assistant fields such as
`refusal`, `audio`, or the legacy `function_call` may be ignored while the
supported text or tool-call fields are converted to Parts. These cases are a
known limitation and should be changed to whole-message `OpaquePart` fallback
before claiming complete Completion message preservation.

## E2E Testing

As of now, `E2E` tests with `tools` are not performed, since there is no public
API for `pytoy_llm.completion` with `tools` arguments.

Codec behavior should be covered by focused unit tests. E2E tests may cover the
public completion API for currently exposed arguments, but tool-enabled E2E
coverage must wait until that public API exists.
