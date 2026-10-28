from importlib.util import module_from_spec, spec_from_file_location
from inspect import isabstract, isclass
from pathlib import Path
from sys import modules
from uuid import uuid4

from pytoy_llm.tool.plugin.contract import LLMPluginTool, LLMPluginTools


class PluginLoadError(RuntimeError):
    """The requested path does not satisfy the plugin loading contract."""


def _load_module(path: Path):
    module_name = f"_pytoy_llm_plugin_{uuid4().hex}"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"Could not create a module loader for {path}")

    module = module_from_spec(spec)
    modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        modules.pop(module_name, None)
        raise
    return module


def _tool_implementations(module) -> tuple[LLMPluginTool, ...]:
    implementations = []
    seen_classes: set[type[LLMPluginTool]] = set()
    for value in vars(module).values():
        if not isclass(value) or value is LLMPluginTool:
            continue
        if value.__module__ != module.__name__ or not issubclass(value, LLMPluginTool):
            continue
        if isabstract(value) or value in seen_classes:
            continue
        seen_classes.add(value)
        implementations.append(value())

    if not implementations:
        raise PluginLoadError(f"No concrete LLMPluginTool subclasses found in {module.__file__}")
    return tuple(implementations)


class FileLoader:
    """Load one Python file and collect its concrete LLMPluginTool subclasses."""

    def load(self, path: Path | str) -> LLMPluginTools:
        plugin_path = Path(path).expanduser().resolve(strict=True)
        if not plugin_path.is_file() or plugin_path.suffix.lower() != ".py":
            raise PluginLoadError(f"Plugin path must be a Python file: {plugin_path}")

        module = _load_module(plugin_path)
        try:
            return LLMPluginTools(_tool_implementations(module))
        except BaseException:
            modules.pop(module.__name__, None)
            raise


class DirectoryLoader:
    """Load Python plugin files directly inside a directory, without recursion."""

    def load(self, path: Path | str) -> LLMPluginTools:
        plugin_dir = Path(path).expanduser().resolve(strict=True)
        if not plugin_dir.is_dir():
            raise PluginLoadError(f"Plugin path must be a directory: {plugin_dir}")

        plugin_files = sorted(
            (
                child
                for child in plugin_dir.iterdir()
                if child.is_file() and child.suffix.lower() == ".py" and child.name != "__init__.py"
            ),
            key=lambda child: child.name,
        )
        if not plugin_files:
            raise PluginLoadError(f"No Python plugin files found directly in {plugin_dir}")

        implementations = tuple(
            tool
            for plugin_file in plugin_files
            for tool in FileLoader().load(plugin_file).plugin_tools
        )
        return LLMPluginTools(implementations)
