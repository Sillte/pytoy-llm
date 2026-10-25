from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import JsonValue

from pytoy_llm.idea import IdeaNote, MetadataSerializationError, TextPosition
from pytoy_llm.idea.domain.metadata import MetaDataProtocol


class FailingMetadata(MetaDataProtocol):
    def __init__(self) -> None:
        self._data: dict[str, Any] = {"key": "value"}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def as_text(self) -> str:
        raise MetadataSerializationError("cannot serialize")

    def as_dict(self) -> dict[str, JsonValue]:
        return self._data

    def clear(self) -> None:
        self._data.clear()


def test_note_parses_front_matter_and_preserves_body_offset() -> None:
    text = "\n---\nid: example\n---\n# Heading\n"

    note = IdeaNote(text=text, path=Path("note.md"), root=Path("."))

    assert note.body == "# Heading\n"
    assert note.body_start_line == 4
    assert note.metadata is not None
    assert note.metadata["id"] == "example"
    assert note.to_text().startswith("---\n")
    assert note.root_folder_path == note.root_directory_path


def test_note_extracts_link_sources_with_source_ranges() -> None:
    note = IdeaNote(
        text="See [target](target.md#L3).",
        path=Path("note.md"),
        root=Path("."),
    )

    assert len(note.link_sources) == 1
    assert note.link_sources[0].start == TextPosition(0, 4)
    assert note.link_sources[0].text_range.end.col > note.link_sources[0].start.col


def test_note_metadata_can_be_added_when_frontmatter_is_absent() -> None:
    note = IdeaNote("Body", Path("note.md"), root=Path("."))
    note.metadata["id"] = "example"

    assert "id: example" in note.text


def test_note_omits_metadata_when_it_cannot_be_serialized(tmp_path: Path) -> None:
    note = IdeaNote.create(
        file_path=tmp_path / "note.md",
        body="Body",
        metadata=FailingMetadata(),
        root=tmp_path,
    )

    assert note.to_text() == "Body"
