from collections.abc import Callable, Sequence
from itertools import chain
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMToolSet(Protocol):
    @property
    def tools(self) -> Sequence[Callable]: ...


type LLMToolsLike = Callable | LLMToolSet | Sequence["LLMToolsLike"]


def from_llm_tools_like(tools: LLMToolsLike) -> Sequence[Callable]:
    if isinstance(tools, LLMToolSet):
        return tools.tools
    if callable(tools):
        return [tools]
    if isinstance(tools, Sequence):
        return list(chain.from_iterable(from_llm_tools_like(elem) for elem in tools))
    assert False, "Implementation Error"
