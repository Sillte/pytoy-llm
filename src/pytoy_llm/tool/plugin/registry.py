from collections.abc import Callable
from pathlib import Path
from typing import Self

from pytoy_llm.tool.plugin.loader import DirectoryLoader, FileLoader, PluginLoadError
from pytoy_llm.tool.plugin.models import LLMPluginCollection


class LLMToolPluginRegistry:
    """Expose the tools and prompt guidance from one loaded plugin collection."""

    def __init__(self, collection: LLMPluginCollection) -> None:
        self._collection = collection

    @classmethod
    def from_any(cls, path: Path | str) -> Self:
        plugin_path = Path(path).expanduser().resolve(strict=True)
        if plugin_path.is_file():
            collection = FileLoader().load(plugin_path)
        elif plugin_path.is_dir():
            collection = DirectoryLoader().load(plugin_path)
        else:
            raise PluginLoadError(f"Plugin path must be a Python file or directory: {plugin_path}")

        return cls(collection=collection)

    @property
    def tools(self) -> tuple[Callable, ...]:
        return self._collection.tools

    @property
    def guidances(self) -> tuple[str, ...]:
        return self._collection.guidances
