from pathlib import Path

import pytest

from pytoy_llm.tools import WorkspaceExplorer
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import (
    FileContent,
    FilePartContent,
    PartialFilesReadResult,
    WorkspaceAccess,
)


def test_inspection_reads_file_beginning_as_partial_content(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("zero\none\ntwo\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    beginning = explorer.inspection.read_text_file("sample.txt", max_lines=2)

    assert isinstance(beginning, FilePartContent)
    assert beginning.content == "zero\none\n"


def test_inspection_reads_requested_file_range(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("zero\none\ntwo\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    selected = explorer.inspection.read_text_file_range("sample.txt", start_line=1, end_line=3)

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


def test_inspection_rejects_outside_paths(tmp_path: Path) -> None:
    (tmp_path.parent / "secret.txt").write_text("top secret\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    outside = explorer.inspection.read_text_file("../secret.txt")

    assert isinstance(outside, ToolError)
    assert outside.kind is ToolErrorKind.INVALID_ARGUMENT


def test_inspection_rejects_directories_as_files(tmp_path: Path) -> None:
    (tmp_path / "folder").mkdir()
    explorer = WorkspaceExplorer.from_any(tmp_path)

    directory = explorer.inspection.read_text_file("folder")

    assert isinstance(directory, ToolError)
    assert directory.kind is ToolErrorKind.INVALID_ARGUMENT


def test_discovery_returns_workspace_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("needle\nother\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    found = explorer.discovery.find_paths("src", patterns="*.py")

    assert not isinstance(found, ToolError)
    assert [item.path for item in found] == ["src/main.py"]


def test_search_returns_workspace_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("needle\nother\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    matches = explorer.search.grep_context("needle", collection_root="src")

    assert not isinstance(matches, ToolError)
    assert len(matches) == 1
    assert matches[0].path == "src/main.py"
    assert matches[0].matches[0].line == 0


def test_empty_excludes_allow_matching_ignored_directory(tmp_path: Path) -> None:
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "config").write_text("needle\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path, excludes=[])

    found = explorer.discovery.find_paths(".", patterns="ignored/*")

    assert not isinstance(found, ToolError)
    assert any(item.path == "ignored/config" for item in found)


def test_inspection_returns_resource_limit_for_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("needle\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path, excludes=[])

    too_large = explorer.inspection.read_text_file("large.txt", max_bytes=1)

    assert isinstance(too_large, ToolError)
    assert too_large.kind is ToolErrorKind.RESOURCE_LIMIT


def test_search_rejects_invalid_regex(tmp_path: Path) -> None:
    (tmp_path / "matches.txt").write_text("needle\nneedle\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    invalid = explorer.search.grep_context("[", regex=True)

    assert isinstance(invalid, ToolError)
    assert invalid.kind is ToolErrorKind.INVALID_ARGUMENT


def test_search_limits_matching_lines(tmp_path: Path) -> None:
    (tmp_path / "matches.txt").write_text("needle\nneedle\n", encoding="utf-8")
    explorer = WorkspaceExplorer.from_any(tmp_path)

    limited = explorer.search.grep_context("needle", max_results=1)

    assert not isinstance(limited, ToolError)
    assert sum(len(context.matches) for context in limited) == 1


def test_gather_file_paths_returns_workspace_relative_paths_and_applies_size_limit(
    tmp_path: Path,
) -> None:
    (tmp_path / "small.txt").write_text("small", encoding="utf-8")
    (tmp_path / "large.txt").write_text("large file", encoding="utf-8")
    access = WorkspaceAccess.from_any(tmp_path)

    paths = access.gather_file_paths(".", ["*.txt"], max_file_bytes=5)

    assert paths == ("small.txt",)


def test_gather_file_paths_allows_unlimited_file_size(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("large file", encoding="utf-8")
    access = WorkspaceAccess.from_any(tmp_path)

    paths = access.gather_file_paths(".", ["*.txt"], max_file_bytes=None)

    assert paths == ("large.txt",)


@pytest.mark.parametrize(
    ("root_name", "expected_kind"),
    [
        ("missing", ToolErrorKind.NOT_FOUND),
        ("file.txt", ToolErrorKind.INVALID_ARGUMENT),
    ],
)
def test_gather_file_paths_returns_error_for_invalid_collection_root(
    tmp_path: Path,
    root_name: str,
    expected_kind: ToolErrorKind,
) -> None:
    access = WorkspaceAccess.from_any(tmp_path)

    if root_name == "file.txt":
        (tmp_path / root_name).write_text("content", encoding="utf-8")

    result = access.gather_file_paths(root_name, ["*"], max_file_bytes=None)

    assert isinstance(result, ToolError)
    assert result.kind is expected_kind
