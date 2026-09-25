from pathlib import Path

from pytoy_llm.tools import WorkspaceExplorer
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import (
    FileContent,
    FilePartContent,
    PartialFilesReadResult,
)


def test_inspection_reads_files_and_ranges(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("zero\none\ntwo\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    beginning = explorer.inspection.read_text_file("sample.txt", max_lines=2)
    selected = explorer.inspection.read_text_file_range("sample.txt", start_line=1, end_line=3)

    assert isinstance(beginning, FilePartContent)
    assert beginning.content == "zero\none\n"
    assert isinstance(selected, FilePartContent)
    assert selected.content == "one\ntwo\n"


def test_inspection_reads_multiple_files_successfully(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("first\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("second\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    result = explorer.inspection.read_text_files(["first.txt", "second.txt"])

    assert isinstance(result, list)
    assert [item.path for item in result] == ["first.txt", "second.txt"]
    assert all(isinstance(item, FileContent) for item in result)


def test_inspection_returns_partial_success_for_batch_read_failures(tmp_path: Path) -> None:
    (tmp_path / "available.txt").write_text("available\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    result = explorer.inspection.read_text_files(["available.txt", "missing.txt"])

    assert isinstance(result, PartialFilesReadResult)
    assert result.status == "partial-success"
    assert [item.path for item in result.successes] == ["available.txt"]
    assert result.failures["missing.txt"].kind is ToolErrorKind.NOT_FOUND


def test_inspection_rejects_invalid_batch_limits(tmp_path: Path) -> None:
    explorer = WorkspaceExplorer.from_any(tmp_path)

    result = explorer.inspection.read_text_files([], max_lines=0)

    assert isinstance(result, ToolError)
    assert result.kind is ToolErrorKind.INVALID_ARGUMENT


def test_inspection_rejects_outside_paths_and_directories(tmp_path: Path) -> None:
    (tmp_path.parent / "secret.txt").write_text("top secret\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    explorer = WorkspaceExplorer.from_any(tmp_path)

    outside = explorer.inspection.read_text_file("../secret.txt")
    directory = explorer.inspection.read_text_file("folder")

    assert isinstance(outside, ToolError)
    assert outside.kind is ToolErrorKind.INVALID_ARGUMENT
    assert isinstance(directory, ToolError)
    assert directory.kind is ToolErrorKind.INVALID_ARGUMENT


def test_discovery_and_search_are_workspace_relative(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("needle\nother\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    found = explorer.discovery.find_paths("src", patterns="*.py")
    matches = explorer.search.grep_context("needle", collection_root="src")

    assert not isinstance(found, ToolError)
    assert [item.path for item in found] == ["src/main.py"]
    assert not isinstance(matches, ToolError)
    assert len(matches) == 1
    assert matches[0].path == "src/main.py"
    assert matches[0].matches[0].line == 0


def test_empty_excludes_are_honored_and_limits_are_structured(tmp_path: Path) -> None:
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "config").write_text("needle\n", encoding="utf-8")
    (tmp_path / "large.txt").write_text("needle\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path, excludes=[])

    found = explorer.discovery.find_paths(".", patterns="ignored/*")
    too_large = explorer.inspection.read_text_file("large.txt", max_bytes=1)

    assert not isinstance(found, ToolError)
    assert any(item.path == "ignored/config" for item in found)
    assert isinstance(too_large, ToolError)
    assert too_large.kind is ToolErrorKind.RESOURCE_LIMIT


def test_search_rejects_invalid_regex_and_limits_matches(tmp_path: Path) -> None:
    (tmp_path / "matches.txt").write_text("needle\nneedle\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    invalid = explorer.search.grep_context("[", regex=True)
    limited = explorer.search.grep_context("needle", max_results=1)

    assert isinstance(invalid, ToolError)
    assert invalid.kind is ToolErrorKind.INVALID_ARGUMENT
    assert not isinstance(limited, ToolError)
    assert sum(len(context.matches) for context in limited) == 1
