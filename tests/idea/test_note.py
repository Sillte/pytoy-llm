from pathlib import Path

from pytoy_llm.idea import IdeaNote, TextPosition


def test_note_parses_front_matter_and_preserves_body_offset() -> None:
    text = "\n---\nid: example\n---\n# Heading\n"

    note = IdeaNote(text=text, path=Path("note.md"), root=Path("."))

    assert note.body == "# Heading\n"
    assert note.body_start_line == 4
    assert note.metadata is not None
    assert note.metadata["id"] == "example"
    assert note.to_text().startswith("---\n")


def test_note_extracts_link_sources_with_source_ranges() -> None:
    note = IdeaNote(
        text="See [target](target.md#L3) and [[other]].",
        path=Path("note.md"),
        root=Path("."),
    )

    assert len(note.link_sources) == 2
    assert note.link_sources[0].start == TextPosition(0, 4)
    assert note.link_sources[0].text_range.end.col > note.link_sources[0].start.col
