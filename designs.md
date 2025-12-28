# Memorandum in the process of design.


## OutputFormat Philosophy

`pytoy-llm` intentionally separates **how the LLM is instructed to generate output**
from **how the final result is returned to the caller**.

This distinction is expressed by the `SyncOutputFormat` abstraction.

### Motivation

There are two different but often conflated concerns when calling an LLM:

1. **Generation contract**  
   How the LLM should format its response  
   (plain text, structured JSON, or a Pydantic model).

2. **Final output type**  
   What the caller actually wants to receive in Python  
   (string, `BaseModel`, custom post-processed object, or raw `ModelResponse`).

In `pytoy-llm`, these concerns are intentionally decoupled.

### SyncOutputFormat

`SyncOutputFormat` represents the *desired final output type*, and derives
the appropriate `response_format` parameter for `litellm.completion`.

Supported input forms:

- `"str"`  
  Return plain text (`str`).

- `"all"`  
  Return the raw `litellm.ModelResponse`.

- `type[BaseModel]`  
  Instruct the LLM to directly generate a structured response
  matching the given Pydantic model.

- `type[CustomLLMOutputModel]`  
  Request a custom, post-processed output type.
  In this case, no `response_format` is passed to LiteLLM,
  and conversion happens after receiving the response.

Internally, all of these are normalized into a `SyncOutputFormat` instance
using `from_any`, ensuring a consistent internal contract.

### Why this design?

- CLI users naturally think in terms of *strings* (`"str"`, `"all"`).
- Python users naturally think in terms of *types* (`BaseModel`, custom classes).
- LiteLLM requires a precise `response_format` decision at call time.

`SyncOutputFormat` acts as a boundary object that keeps these concerns cleanly separated.


## OutputConverter Responsibility

`OutputConverter` is responsible for **transforming a `litellm.ModelResponse`
into the final output type requested by the caller**.

It deliberately does **not**:

- Decide how the LLM should generate output
- Manage connection configuration
- Store or manage conversation state

Those responsibilities belong to other components.

### Conversion Rules

Given a `litellm.ModelResponse` and a resolved `SyncOutputFormat`:

- If the requested output type is `str`  
  → Extract and return the primary text content.

- If the requested output type is `litellm.ModelResponse`  
  → Return the response as-is.

- If the requested output type is a `BaseModel`  
  → Parse the structured response into that model.

- If the requested output type is a `CustomLLMOutputModel`  
  → Delegate to `CustomLLMOutputModel.from_litellm_model_response`.

This strict separation allows:

- Clear reasoning about data flow
- Easier testing
- Independent evolution of output formats and generation logic
