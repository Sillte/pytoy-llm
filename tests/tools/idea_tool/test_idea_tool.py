from pathlib import Path

from pytoy_llm.idea import IdeaSpace
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.idea_tool import IdeaTool


def test_get_metadata_returns_metadata_and_none_for_invalid_paths(tmp_path: Path) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("---\nstatus: active\ntags: [one, two]\n---\nBody\n")
    (tmp_path / "folder").mkdir()

    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.get_metadata(["note.md", "missing.md", "folder"])

    assert result == {
        "note.md": {"status": "active", "tags": ["one", "two"]},
        "missing.md": None,
        "folder": None,
    }


def test_create_subspace_creates_a_new_directory(tmp_path: Path) -> None:
    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.create_subspace("knowledge")

    assert result == "knowledge"
    assert (tmp_path / "knowledge").is_dir()


def test_create_subspace_rejects_existing_directory(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.create_subspace("knowledge")

    assert isinstance(result, ToolError)
    assert result.kind == ToolErrorKind.INVALID_ARGUMENT


def test_delete_subspace_deletes_only_empty_directories(tmp_path: Path) -> None:
    (tmp_path / "knowledge").mkdir()
    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.delete_subspace("knowledge")

    assert result == "knowledge"
    assert not (tmp_path / "knowledge").exists()


def test_delete_subspace_rejects_non_empty_directories(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "note.md").write_text("note", encoding="utf-8")
    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.delete_subspace("knowledge")

    assert isinstance(result, ToolError)
    assert result.kind == ToolErrorKind.INVALID_ARGUMENT
    assert knowledge.exists()


def test_get_convention_pivot_paths_returns_root_and_nested_conventions(
    tmp_path: Path,
) -> None:
    (tmp_path / ".convention.md").write_text("root convention\n")
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / ".idea_space_convention.md").write_text("knowledge convention\n")
    (tmp_path / "knowledge" / "python").mkdir()
    (tmp_path / "knowledge" / "python" / ".convention.md").write_text("python convention\n")

    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.get_convention_pivot_paths()

    assert result == [".", "knowledge", "knowledge/python"]


def test_get_convention_pivot_paths_represents_root_as_dot(tmp_path: Path) -> None:
    (tmp_path / ".convention.md").write_text("root convention\n")
    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.get_convention_pivot_paths()

    assert result == ["."]


def test_get_convention_pivot_paths_returns_empty_sequence_without_conventions(
    tmp_path: Path,
) -> None:
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "note.md").write_text("note\n")
    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.get_convention_pivot_paths()

    assert result == []


def test_update_metadata_preserves_body_and_merges_metadata(tmp_path: Path) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("---\nstatus: draft\nowner: alice\n---\n# Body\n")

    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.update_metadata("note.md", {"status": "published", "tags": ["one"]})

    assert result == "note.md"
    assert (
        note_path.read_text() == "---\nstatus: published\nowner: alice\ntags:\n- one\n---\n# Body\n"
    )


def test_update_metadata_returns_error_for_missing_note(tmp_path: Path) -> None:
    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.update_metadata("missing.md", {"status": "published"})

    assert isinstance(result, ToolError)


def test_update_metadata_clear_replaces_existing_metadata(tmp_path: Path) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("---\nstatus: draft\nowner: alice\n---\n# Body\n")

    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.update_metadata("note.md", {"status": "published"}, clear=True)

    assert result == "note.md"
    assert note_path.read_text() == "---\nstatus: published\n---\n# Body\n"


def test_update_metadata_clear_with_empty_metadata_removes_frontmatter(tmp_path: Path) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("---\nstatus: draft\n---\n# Body\n")

    tool = IdeaTool(IdeaSpace(tmp_path))

    result = tool.update_metadata("note.md", {}, clear=True)

    assert result == "note.md"
    assert note_path.read_text() == "# Body\n"
