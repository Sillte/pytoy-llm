import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from git import Diff
from pytoy_llm.materials.git_diffs.collectors import GitDiffCollector, FileAddCreator, FileDeleteCreator, FileModifyCreator
from pytoy_llm.materials.git_diffs.models import FileAdd, FileDelete, FileModify, AtomicChange, LineRange, GitDiffBundleQuery

# --- FileAddCreator テスト ---
def test_file_add_creator():
    mock_diff = MagicMock()
    mock_diff.diff = b"+++ b/foo.txt\n+line1\n+line2\n"
    mock_diff.b_path = "foo.txt"

    creator = FileAddCreator()
    result = creator(mock_diff)

    assert isinstance(result, FileAdd)
    assert result.path == Path("foo.txt")
    assert result.lines == ["line1", "line2"]

# --- FileDeleteCreator テスト ---
def test_file_delete_creator():
    mock_diff = MagicMock()
    mock_diff.diff = b"--- a/foo.txt\n-line1\n-line2\n"
    mock_diff.a_path = "foo.txt"

    creator = FileDeleteCreator()
    result = creator(mock_diff)

    assert isinstance(result, FileDelete)
    assert result.path == Path("foo.txt")
    assert result.old_lines == ["line1", "line2"]

# --- FileModifyCreator テスト ---
def test_file_modify_creator():
    mock_diff = MagicMock()
    mock_diff.diff = (
        b"@@ -1,2 +1,2 @@\n"
        b"-old1\n+new1\n"
        b"-old2\n+new2\n"
    )
    mock_diff.a_path = "foo.txt"
    mock_diff.b_path = "foo.txt"

    creator = FileModifyCreator()
    result = creator(mock_diff)

    assert isinstance(result, FileModify)
    assert result.path == Path("foo.txt")
    assert len(result.atomic_changes) == 1
    ac = result.atomic_changes[0]
    assert ac.old_lines == ["old1", "old2"]
    assert ac.new_lines == ["new1", "new2"]
    assert ac.range.start == 0
    assert ac.range.end == 2

# --- GitDiffCollector テスト ---
@patch("pytoy_llm.materials.git_diffs.collectors.Repo")
def test_git_diff_collector(mock_repo_cls):
    # モック Repo と diff
    mock_repo = MagicMock()
    mock_repo_cls.return_value = mock_repo

    # コミットモック
    mock_diff = MagicMock()
    mock_diff.change_type = "A"
    mock_diff.b_path = "foo.txt"
    mock_diff.diff = b"+++ b/foo.txt\n+line1\n"
    mock_repo.commit.return_value.diff.return_value = [mock_diff]
    mock_repo.commit.return_value.committed_date = 1234567890
    mock_repo.git_dir = "."

    collector = GitDiffCollector(repo_path=".")
    query = GitDiffBundleQuery(from_rev="abc", to_rev="def")
    diff_container = collector.get_bundle(query)

    assert diff_container.root_location == collector.root_location
    assert len(diff_container.file_diffs) == 1
    fd = diff_container.file_diffs[0]
    assert isinstance(fd.operation, FileAdd)
    assert fd.operation.lines == ["line1"]
    assert fd.timestamp == 1234567890
