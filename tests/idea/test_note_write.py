from pathlib import Path

from pytoy_llm.idea import FileWriterProtocol, IdeaNote
from pytoy_llm.idea.domain.uri import Uri


class MemoryFileWriter(FileWriterProtocol):
    def __init__(self) -> None:
        self.writes: list[tuple[Path, str]] = []

    def write(self, text: str, file_path: Path) -> None:
        self.writes.append((file_path, text))


def test_note_writes_serialized_text_through_file_writer(tmp_path: Path) -> None:
    note = IdeaNote(
        text="---\nid: example\n---\nBody\n",
        path=Path("note.md"),
        root=tmp_path,
    )
    writer = MemoryFileWriter()

    note.write(writer)

    assert writer.writes == [(tmp_path / "note.md", note.to_text())]


def test_note_writes_to_disk_by_default(tmp_path: Path) -> None:
    note_path = tmp_path / "note.md"
    note = IdeaNote(text="Body\n", path=note_path, root=tmp_path)

    note.write()

    assert note_path.read_text(encoding="utf8") == "Body\n"


def test_link_sources_reflect_updated_body() -> None:
    note = IdeaNote(text="[old](old.md)", path=Path("note.md"), root=Path("."))

    assert [link.target for link in note.link_sources] == [Uri.from_any("old.md")]

    note.set_body("[new](new.md)")

    assert [link.target for link in note.link_sources] == [Uri.from_any("new.md")]
