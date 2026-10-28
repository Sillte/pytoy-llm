from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass


class LLMPluginTool(ABC):
    """Extension contract for one tool implementation."""

    @property
    @abstractmethod
    def tool(self) -> Callable:
        """Return the callable registered with the LLM agent."""
        raise NotImplementedError


@dataclass(frozen=True, init=False)
class LLMPluginTools:
    """An immutable collection of plugin implementations and their callables."""

    _plugin_tools: tuple[LLMPluginTool, ...]
    _tools: tuple[Callable, ...]

    def __init__(self, plugin_tools: Iterable[LLMPluginTool] = ()) -> None:
        implementations = tuple(plugin_tools)
        if not all(isinstance(item, LLMPluginTool) for item in implementations):
            raise TypeError("Every plugin tool must inherit from LLMPluginTool.")

        tools = tuple(item.tool for item in implementations)
        if not all(callable(tool) for tool in tools):
            raise TypeError("Every LLMPluginTool.tool must return a callable.")

        object.__setattr__(self, "_plugin_tools", implementations)
        object.__setattr__(self, "_tools", tools)

    @property
    def plugin_tools(self) -> tuple[LLMPluginTool, ...]:
        return self._plugin_tools

    @property
    def tools(self) -> tuple[Callable, ...]:
        return self._tools
