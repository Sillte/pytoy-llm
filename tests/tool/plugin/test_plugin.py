from pathlib import Path

import pytest

from pytoy_llm.models.llm_tools import from_llm_tools_like
from pytoy_llm.tool.plugin import (
    DirectoryLoader,
    FileLoader,
    LLMPluginTool,
    LLMPluginTools,
    LLMToolPluginRegistry,
    PluginLoadError,
)


class _CallableTool(LLMPluginTool):
    def __init__(self, callable_tool):
        self._callable_tool = callable_tool

    @property
    def tool(self):
        return self._callable_tool


def _plugin_source(class_name: str, tool_name: str, result: str) -> str:
    return (
        "from pytoy_llm.tool.plugin import LLMPluginTool\n"
        f"class {class_name}(LLMPluginTool):\n"
        "    @property\n"
        "    def tool(self):\n"
        f"        def {tool_name}() -> str:\n"
        f"            return {result!r}\n"
        f"        return {tool_name}\n"
    )


def test_file_loader_discovers_concrete_tool_subclasses(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text(_plugin_source("ExampleTool", "example_tool", "ok"), encoding="utf-8")

    plugin_tools = FileLoader().load(plugin_path)

    assert len(plugin_tools.plugin_tools) == 1
    assert plugin_tools.tools[0]() == "ok"


def test_file_loader_rejects_module_without_tool_subclasses(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(PluginLoadError, match="No concrete LLMPluginTool"):
        FileLoader().load(plugin_path)


def test_directory_loader_reads_only_direct_python_files_in_order(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text(_plugin_source("BTool", "b_tool", "b"), encoding="utf-8")
    (tmp_path / "a.py").write_text(_plugin_source("ATool", "a_tool", "a"), encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.py").write_text(
        _plugin_source("NestedTool", "nested_tool", "nested"), encoding="utf-8"
    )

    plugin_tools = DirectoryLoader().load(tmp_path)

    assert [tool.__name__ for tool in plugin_tools.tools] == ["a_tool", "b_tool"]
    assert [tool() for tool in plugin_tools.tools] == ["a", "b"]


def test_registry_exposes_tools_from_loaded_file(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text(_plugin_source("ExampleTool", "example_tool", "ok"), encoding="utf-8")
    registry = LLMToolPluginRegistry()

    plugin_tools = registry.load(plugin_path)

    assert isinstance(plugin_tools, LLMPluginTools)
    assert registry.plugin_groups == (plugin_tools,)
    assert registry.tools == plugin_tools.tools
    assert from_llm_tools_like(registry) == registry.tools


def test_registry_aggregates_tools_from_file_and_directory(tmp_path: Path) -> None:
    plugin_path = tmp_path / "one.py"
    plugin_path.write_text(_plugin_source("OneTool", "one_tool", "one"), encoding="utf-8")
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "two.py").write_text(
        _plugin_source("TwoTool", "two_tool", "two"), encoding="utf-8"
    )
    registry = LLMToolPluginRegistry()

    registry.load(plugin_path)
    registry.load(plugin_dir)

    assert [tool() for tool in registry.tools] == ["one", "two"]
    assert from_llm_tools_like(registry) == registry.tools


def test_registry_rejects_lambda_tools() -> None:
    registry = LLMToolPluginRegistry()
    plugin_tools = LLMPluginTools([_CallableTool(lambda: "anonymous")])

    with pytest.raises(ValueError, match="non-lambda callable name"):
        registry.register(plugin_tools)
    assert registry.plugin_groups == ()


def test_registry_rejects_duplicate_tool_names() -> None:
    def make_tool():
        def duplicate_tool() -> str:
            return "ok"

        return duplicate_tool

    registry = LLMToolPluginRegistry()
    registry.register(LLMPluginTools([_CallableTool(make_tool())]))

    with pytest.raises(ValueError, match="Duplicate plugin tool name"):
        registry.register(LLMPluginTools([_CallableTool(make_tool())]))
    assert len(registry.plugin_groups) == 1


def test_plugin_tools_rejects_non_contract_implementations() -> None:
    with pytest.raises(TypeError, match="inherit from LLMPluginTool"):
        LLMPluginTools([object()])
