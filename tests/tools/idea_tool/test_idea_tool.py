from pathlib import Path

from pytoy_llm.idea import IdeaSpace
from pytoy_llm.tools.errors import ToolError
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
