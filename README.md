# pytoy-llm

This project is currently a personal project rather than a public release.
For now, it focuses on exploring software architecture, with a particular focus on integration with Vim plugin.

## Principles

- Stateless LLM usage
    - Structured system prompt construction
    - Typed contracts
    - OpenAI SDK for completion by default, with optional LiteLLM; PydanticAI for agents

- Task mechanism with state management
    - Session management
    - Event-driven activity logs

This library is primarily intended to be used by [`vim-pytoy`](https://github.com/Sillte/vim-pytoy).

---

## Usage

### Configuration

First, run the following script to generate a connection file:

```python
from pytoy_llm import initialize_configuration

path = initialize_configuration("first_connect")
print("Connection file:", path)
```

The script will print the path to a configuration file like the following.
Fill in your API information and the model you want to use.

```json
{
    "model": "",
    "base_url": "",
    "api_key": ""
}
```

For example:

```json
{
    "model": "gemini/gemini-2.5-flash",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "api_key": "YOUR_SECRET_KEY"
}
```

`completion` uses the OpenAI Chat Completions protocol by default, so `base_url` must expose an OpenAI-compatible endpoint. For example, Ollama uses `http://localhost:11434/v1`. To opt into LiteLLM's provider support, install `pytoy-llm[litellm]` and set `"completion_backend": "litellm"` in the connection file. Existing connection files default to the OpenAI backend. `LLMFacade` accepts a `CompletionClientFactory` through `completion_client_factory` for dependency injection. The `responses` protocol is not supported by `completion` yet; `run` continues to use PydanticAI's provider integrations.

### `completion` — Stateless usage

The simplest usage is:

* Input: `str`
* Output: `str`

```python
from pytoy_llm import completion

output = completion(
    "Hello, there.",
    output_type=str,
    connection="first_connect",
)
```

### `Task` — Stateful usage

For examples, refer to the [`usage`](./usage) folder.

---

## Security Notice

**Security consideration:** Tools may expose workspace contents to an LLM.
When using `LLMToolsLike`, carefully review which tools are enabled and what data they can access.

The `pytoy_llm/tools` package includes tools that can be made available to LLMs.

### `WorkspaceExplorer`

* The workspace corresponds to the root of the package.
* This tool only gathers files and folders under the specified workspace.
* The gathered contents may be sent to the LLM.

