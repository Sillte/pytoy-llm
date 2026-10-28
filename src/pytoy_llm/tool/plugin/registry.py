from collections.abc import Callable
from pathlib import Path

from pytoy_llm.tool.plugin.contract import LLMPluginTools
from pytoy_llm.tool.plugin.loader import DirectoryLoader, FileLoader, PluginLoadError


class LLMToolPluginRegistry:
    """Collect loaded plugin tool groups and expose their callables."""

    def __init__(self) -> None:
        self._plugin_groups: list[LLMPluginTools] = []

    def register(self, plugin_tools: LLMPluginTools) -> None:
        if not isinstance(plugin_tools, LLMPluginTools):
            raise TypeError("Only LLMPluginTools instances can be registered.")

        names: set[str] = set()
        tools = (tool for group in (*self._plugin_groups, plugin_tools) for tool in group.tools)
        for tool in tools:
            name = getattr(tool, "__name__", None)
            if not isinstance(name, str) or not name.strip() or name == "<lambda>":
                raise ValueError("Every plugin tool must have a non-lambda callable name.")
            if name in names:
                raise ValueError(f"Duplicate plugin tool name: {name!r}")
            names.add(name)

        self._plugin_groups.append(plugin_tools)

    def load(self, path: Path | str) -> LLMPluginTools:
        plugin_path = Path(path).expanduser().resolve(strict=True)
        if plugin_path.is_file():
            plugin_tools = FileLoader().load(plugin_path)
        elif plugin_path.is_dir():
            plugin_tools = DirectoryLoader().load(plugin_path)
        else:
            raise PluginLoadError(f"Plugin path must be a Python file or directory: {plugin_path}")

        self.register(plugin_tools)
        return plugin_tools

    @property
    def plugin_groups(self) -> tuple[LLMPluginTools, ...]:
        return tuple(self._plugin_groups)

    @property
    def tools(self) -> tuple[Callable, ...]:
        return tuple(tool for group in self._plugin_groups for tool in group.tools)
