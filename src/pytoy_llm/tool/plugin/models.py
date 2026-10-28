from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Self

from pytoy_llm.models.llm_tools import from_llm_tools_like
from pytoy_llm.tool.plugin.contract import LLMToolsPlugin


def _normalize_guidance(guidance: str | None) -> str | None:
    if guidance is not None and not isinstance(guidance, str):
        raise TypeError("Plugin guidance must be a string or None.")
    normalized = guidance.strip() if guidance is not None else None
    return normalized or None


@dataclass(frozen=True)
class LLMPluginCollection:
    """Loaded plugin tools and guidance for possible system-prompt inclusion."""

    plugins: tuple[LLMToolsPlugin, ...]
    tools: tuple[Callable, ...]
    guidances: tuple[str, ...]

    @classmethod
    def from_plugins(cls, plugins: Iterable[LLMToolsPlugin]) -> Self:
        plugin_items = tuple(plugins)
        if not all(isinstance(plugin, LLMToolsPlugin) for plugin in plugin_items):
            raise TypeError("Every plugin must inherit from LLMToolsPlugin.")

        try:
            tools = tuple(
                tool for plugin in plugin_items for tool in from_llm_tools_like(plugin.tools)
            )
        except AssertionError as error:
            raise TypeError(
                "Plugin tools must be callables, tool sets, or sequences of those."
            ) from error

        names: set[str] = set()
        for tool in tools:
            name = getattr(tool, "__name__", None)
            if not isinstance(name, str) or not name.strip() or name == "<lambda>":
                raise ValueError("Every plugin tool must have a non-lambda callable name.")
            if name in names:
                raise ValueError(f"Duplicate plugin tool name: {name!r}")
            names.add(name)

        guidances = tuple(
            normalized
            for plugin in plugin_items
            if (normalized := _normalize_guidance(plugin.guidance())) is not None
        )
        return cls(plugins=plugin_items, tools=tools, guidances=guidances)
