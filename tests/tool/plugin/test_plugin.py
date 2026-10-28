import sys
from pathlib import Path

import pytest

from pytoy_llm.models.llm_tools import from_llm_tools_like
from pytoy_llm.tool.plugin import (
    DirectoryLoader,
    FileLoader,
    LLMToolPluginRegistry,
    PluginLoadError,
)


def _plugin_source(
    class_name: str,
    tool_name: str,
    result: str,
    *,
    guidance: str | None = None,
) -> str:
    source = [
        "from pytoy_llm.tool.plugin import LLMToolsPlugin",
        f"class {class_name}(LLMToolsPlugin):",
        "    @property",
        "    def tools(self):",
        f"        def {tool_name}() -> str:",
        f"            return {result!r}",
        f"        return [{tool_name}]",
    ]
    if guidance is not None:
        source.extend(
            [
                "    def guidance(self) -> str | None:",
                f"        return {guidance!r}",
            ]
        )
    return "\n".join(source) + "\n"


def test_file_loader_discovers_concrete_plugin_sets(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text(_plugin_source("ExamplePlugin", "example_tool", "ok"), encoding="utf-8")

    plugin_tools = FileLoader().load(plugin_path)

    assert len(plugin_tools.plugins) == 1
    assert plugin_tools.tools[0]() == "ok"
    assert plugin_tools.guidances == ()


def test_file_loader_reuses_module_but_creates_fresh_plugin_instances(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text(_plugin_source("ExamplePlugin", "example_tool", "ok"), encoding="utf-8")
    loader = FileLoader()

    first = loader.load(plugin_path)
    second = loader.load(plugin_path)

    assert second is not first
    assert first.plugins[0] is not second.plugins[0]
    assert type(first.plugins[0]) is type(second.plugins[0])
    assert type(first.plugins[0]).__module__ == type(second.plugins[0]).__module__


def test_file_loader_reloads_changed_source_and_prunes_old_module(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text(_plugin_source("ExamplePlugin", "example_tool", "old"), encoding="utf-8")
    loader = FileLoader()
    first = loader.load(plugin_path)
    old_module_name = type(first.plugins[0]).__module__

    plugin_path.write_text(_plugin_source("ExamplePlugin", "example_tool", "new"), encoding="utf-8")
    second = loader.load(plugin_path)

    assert first.tools[0]() == "old"
    assert second.tools[0]() == "new"
    assert old_module_name not in sys.modules
    modules_for_path = [
        module
        for module in sys.modules.values()
        if getattr(module, "__file__", None) == str(plugin_path.resolve())
    ]
    assert len(modules_for_path) == 1


def test_failed_plugin_update_keeps_previous_module_cached(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text(_plugin_source("ExamplePlugin", "example_tool", "old"), encoding="utf-8")
    loader = FileLoader()
    original = loader.load(plugin_path)
    original_module_name = type(original.plugins[0]).__module__
    plugin_path.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(PluginLoadError, match="No concrete LLMToolsPlugin"):
        loader.load(plugin_path)

    assert original_module_name in sys.modules
    assert original.tools[0]() == "old"


def test_identical_plugin_sources_at_different_paths_are_not_shared(tmp_path: Path) -> None:
    source = _plugin_source("ExamplePlugin", "example_tool", "ok")
    first_path = tmp_path / "first" / "plugin.py"
    second_path = tmp_path / "second" / "plugin.py"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text(source, encoding="utf-8")
    second_path.write_text(source, encoding="utf-8")

    first = FileLoader().load(first_path)
    second = FileLoader().load(second_path)

    assert first is not second
    assert type(first.plugins[0]).__module__ != type(second.plugins[0]).__module__


def test_file_loader_collects_plugin_guidance(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text(
        _plugin_source(
            "ExamplePlugin",
            "example_tool",
            "ok",
            guidance="Use this plugin for project-level tasks.",
        ),
        encoding="utf-8",
    )

    plugin_tools = FileLoader().load(plugin_path)

    assert plugin_tools.guidances == ("Use this plugin for project-level tasks.",)


def test_file_loader_rejects_module_without_plugin_contract(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(PluginLoadError, match="No concrete LLMToolsPlugin"):
        FileLoader().load(plugin_path)


def test_directory_loader_reads_only_direct_python_files_in_order(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text(
        _plugin_source("BPlugin", "b_tool", "b", guidance="B guidance"), encoding="utf-8"
    )
    (tmp_path / "a.py").write_text(
        _plugin_source("APlugin", "a_tool", "a", guidance="A guidance"), encoding="utf-8"
    )
    (tmp_path / "helpers.py").write_text("def helper():\n    return 'helper'\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.py").write_text(
        _plugin_source("NestedPlugin", "nested_tool", "nested"), encoding="utf-8"
    )

    plugin_tools = DirectoryLoader().load(tmp_path)

    assert [getattr(tool, "__name__", None) for tool in plugin_tools.tools] == [
        "a_tool",
        "b_tool",
    ]
    assert [tool() for tool in plugin_tools.tools] == ["a", "b"]
    assert plugin_tools.guidances == ("A guidance", "B guidance")


def test_directory_loader_supports_relative_imports_and_helper_updates(tmp_path: Path) -> None:
    helper_path = tmp_path / "helpers.py"
    helper_path.write_text(
        "def create_tool():\n"
        "    def helper_tool() -> str:\n"
        "        return 'first'\n"
        "    return helper_tool\n",
        encoding="utf-8",
    )
    (tmp_path / "plugin.py").write_text(
        "from pytoy_llm.tool.plugin import LLMToolsPlugin\n"
        "from .helpers import create_tool\n"
        "class ExamplePlugin(LLMToolsPlugin):\n"
        "    @property\n"
        "    def tools(self):\n"
        "        return [create_tool()]\n",
        encoding="utf-8",
    )
    loader = DirectoryLoader()

    first = loader.load(tmp_path)
    old_package_name = type(first.plugins[0]).__module__.rpartition(".")[0]
    same_source = loader.load(tmp_path)
    assert same_source is not first
    assert same_source.plugins[0] is not first.plugins[0]
    assert type(same_source.plugins[0]) is type(first.plugins[0])
    helper_path.write_text(
        "def create_tool():\n"
        "    def helper_tool() -> str:\n"
        "        return 'updated'\n"
        "    return helper_tool\n",
        encoding="utf-8",
    )
    second = loader.load(tmp_path)

    assert first.tools[0]() == "first"
    assert second.tools[0]() == "updated"
    assert old_package_name not in sys.modules


def test_directory_loader_rejects_directory_without_plugins(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text("value = 1\n", encoding="utf-8")

    with pytest.raises(PluginLoadError, match="No concrete LLMToolsPlugin"):
        DirectoryLoader().load(tmp_path)


def test_registry_exposes_tools_and_guidance_from_loaded_file(tmp_path: Path) -> None:
    plugin_path = tmp_path / "plugin.py"
    plugin_path.write_text(
        _plugin_source(
            "ExamplePlugin",
            "example_tool",
            "ok",
            guidance="Use this plugin for one item.",
        ),
        encoding="utf-8",
    )

    registry = LLMToolPluginRegistry.from_any(plugin_path)

    assert registry.tools[0]() == "ok"
    assert registry.guidances == ("Use this plugin for one item.",)
    assert from_llm_tools_like(registry) == registry.tools


def test_registry_loads_a_directory(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text(
        _plugin_source("OnePlugin", "one_tool", "one"), encoding="utf-8"
    )
    (tmp_path / "two.py").write_text(
        _plugin_source("TwoPlugin", "two_tool", "two"), encoding="utf-8"
    )

    registry = LLMToolPluginRegistry.from_any(tmp_path)

    assert [tool() for tool in registry.tools] == ["one", "two"]
    assert from_llm_tools_like(registry) == registry.tools
