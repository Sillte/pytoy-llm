from pathlib import Path

from pytoy_llm.idea import IdeaNote, IdeaSpace


def test_space_lists_notes_recursively_in_path_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")

    notes = IdeaSpace(tmp_path).get_notes(depth=None)

    assert [note.path for note in notes] == ["b.md", "nested/a.md"]
    assert all(isinstance(note, IdeaNote) for note in notes)


def test_space_only_treats_markdown_files_as_notes_in_subspaces(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "note.txt").write_text("note", encoding="utf-8")
    (nested / "note.md").write_text("note", encoding="utf-8")

    space = IdeaSpace(tmp_path)

    assert [note.path for note in space.get_subspaces()[0].get_notes()] == ["nested/note.md"]
