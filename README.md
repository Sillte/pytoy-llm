# pytoy-llm

A thin, stateless LLM CLI wrapper designed for editor integrations  
(Vim / Neovim).  
Essentially, this library is LLM used for  [`vim-pytoy`](https://github.com/Sillte/vim-pytoy).
 
`pytoy-llm` provides a **minimal, provider-agnostic boundary** around LLM
completion APIs using [`litellm`](https://github.com/BerriAI/litellm),
with strong emphasis on:

- Stateless execution
- Explicit message passing
- Typed contracts (via Pydantic) (Not yet.)

---

## Design Philosophy

`pytoy-llm` intentionally avoids being “smart”.

It does **not**:
- Manage conversation state
- Store chat history
- Perform retries or re-asking
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

---

## Why a CLI?

Vim / Neovim integrations often rely on job-based execution.
A CLI boundary ensures:

- No shared memory assumptions
- Clear stdin / stdout contracts
- Compatibility with restricted environments
- Clean separation between editor logic and LLM logic

The CLI is designed to be **machine-facing**, not human-interactive.

---

## Installation

```bash
pip install pytoy-llm
