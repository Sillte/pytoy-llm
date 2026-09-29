from pathlib import Path

from pytoy_llm.idea import IdeaNote, IdeaSpace


def test_space_lists_notes_recursively_in_path_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")

    notes = IdeaSpace(tmp_path).get_notes(depth=None)

    assert [note.idea_path for note in notes] == ["b.md", "nested/a.md"]
    assert all(isinstance(note, IdeaNote) for note in notes)


def test_subspace_lists_only_markdown_notes(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "note.txt").write_text("note", encoding="utf-8")
    (nested / "note.md").write_text("note", encoding="utf-8")

    space = IdeaSpace(tmp_path)

    assert [note.idea_path for note in space.get_subspaces()[0].get_notes()] == ["nested/note.md"]


def test_root_space_returns_root_level_space_without_creating_it(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    space = IdeaSpace.from_path(nested, root=tmp_path, with_creation=True)

    root_space = space.root_space

    assert root_space.root_directory_path == tmp_path
    assert root_space.directory_path == tmp_path
    assert root_space.root_folder_path == root_space.root_directory_path
    assert root_space.folder_path == root_space.directory_path
    assert root_space.path == "."
    assert nested.exists()
    assert not (tmp_path / IdeaSpace.ROOT_MARKER_FILE_NAME).exists()
