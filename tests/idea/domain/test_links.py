from pathlib import Path

from pytoy_llm.idea import (
    IdeaLink,
    LineLocation,
    LinkReachabilityChecker,
    TextPosition,
    TextRange,
)


def make_idea_link(uri: str) -> IdeaLink:
    text_range = TextRange(TextPosition(0, 0), TextPosition(0, 1))
    return IdeaLink(source_path=Path("source.md"), source_text_range=text_range, uri=uri)


def test_local_path_is_converted_to_a_standard_file_uri(tmp_path: Path) -> None:
    target_path = tmp_path / "notes" / "hello world.md"
    target_path.parent.mkdir()
    target_path.touch()
    link = IdeaLink(
        source_path=tmp_path / "source.md",
        source_text_range=TextRange(TextPosition(0, 0), TextPosition(0, 1)),
        uri=target_path.as_uri(),
        target_location=LineLocation(),
    )

    assert link.target_file_path == target_path
    assert LinkReachabilityChecker().check(link) is True


def test_file_uri_is_converted_back_to_a_path() -> None:
    link = make_idea_link("file:///C:/notes/hello%20world.md")

    assert link.target_file_path
    assert link.target_file_path == Path("C:/notes/hello world.md")
    assert LinkReachabilityChecker().check(link) is False


def test_file_url_is_recovered_back_to_a_path() -> None:
    link = make_idea_link("file:///tmp/hello%20world.md")
    assert link.target_file_path
    assert link.target_file_path == Path("/tmp/hello world.md")
    assert LinkReachabilityChecker().check(link) is False


def test_remote_uri_is_not_treated_as_a_file_path() -> None:
    link = make_idea_link("https://example.com/notes.md")

    assert link.target_file_path is None
