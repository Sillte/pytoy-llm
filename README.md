**This library is prototypical and not sufficiently tested.**


# pytoy-llm

A thin, stateless LLM CLI wrapper designed for editor integrations  
(Vim / Neovim).

This library is primarily used by [`vim-pytoy`](https://github.com/Sillte/vim-pytoy).
 
`pytoy-llm` provides a **minimal, provider-agnostic boundary** around LLM
completion APIs using [`litellm`](https://github.com/BerriAI/litellm),
with strong emphasis on:

- Stateless execution
- Typed contracts (via Pydantic, partially implemented)

---

## Design Philosophy

`pytoy-llm` intentionally avoids being “smart”.

It does **not**:
- Manage conversation state
- Store chat history
- Own prompt construction logic
- Act as an agent framework

Instead, it acts as a **pure execution boundary**:

> “Given messages and a connection, return a completion.”

All state, orchestration, and higher-level logic are expected to live
**outside** this CLI — for example, in a Vim plugin or a Python library.

This design keeps the tool:
- Predictable
- Easy to audit
- Easy to replace or extend

### Usages

#### Configuration setting

First, execute the following script to generate a connection file.
```python
from pytoy_llm import initialize_configuration
path = initialize_configuration("first_connect")
print("ConnectionFile:", path) 
```
You will see the path to a configuration file like the following.
Please fill your API info and model you want to use there.  
```
{
    "model": "",
    "base_url": "",
    "api_key": ""
}
```

Example: 
```
{
    "model": "gemini/gemini-2.5-flash",
    "base_url": "https://generativelanguage.googleapis.com/v1beta",
    "api_key": "<YOUR_SECRET_KEY>"
}
```

#### `litellm.completion`.

The simplest usage:

- Input: `str`
- Output: `str`

```
from pytoy_llm import completion
output = completion("Hello, there.", output_format="str", connection="first_connect")
```

More advanced output formats (e.g. Pydantic models) are supported and documented...

(To be continue...)
