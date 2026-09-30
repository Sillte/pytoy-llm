from pathlib import Path

from pytoy_llm.idea import (
    IdeaGraph,
    IdeaLink,
    IdeaNote,
    IdeaSpace,
    LineLocation,
    UnresolvedIdeaLink,
)
from pytoy_llm.idea.domain.uri import Uri


def test_graph_resolves_markdown_links(tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    target_path = tmp_path / "target file.md"
    source_path.write_text("[target](target%20file.md#L3)", encoding="utf-8")
    target_path.write_text("target", encoding="utf-8")
    source = IdeaNote.from_path(source_path, root=tmp_path)

    links = IdeaGraph(IdeaSpace(tmp_path)).resolve_links(source)

    assert len(links) == 1
    assert isinstance(links[0], IdeaLink)
    assert links[0].uri == Uri.from_any(target_path.as_uri())
    assert links[0].target_location == LineLocation(line=2)


def test_graph_returns_unresolved_link_for_target_outside_root(tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    source_path.write_text("[outside](../outside.md)", encoding="utf-8")
    source = IdeaNote.from_path(source_path, root=tmp_path)

    links = IdeaGraph(IdeaSpace(tmp_path)).resolve_links(source, only_valid=False)

    assert len(links) == 1
    assert isinstance(links[0], UnresolvedIdeaLink)


def test_graph_returns_unresolved_link_for_retired_repo_scheme(tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    source_path.write_text("[unknown](unknown://project/notes.md)", encoding="utf-8")
    source = IdeaNote.from_path(source_path, root=tmp_path)

    links = IdeaGraph(IdeaSpace(tmp_path)).resolve_links(source, only_valid=False)

    assert len(links) == 1
    assert len(links) == 1
    assert isinstance(links[0], IdeaLink)
    assert links[0].uri.scheme == "unknown"
    assert links[0].uri.authority == "project"


def test_graph_finds_backlinks(tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    target_path = tmp_path / "target.md"
    source_path.write_text("[target](target.md)", encoding="utf-8")
    target_path.write_text("target", encoding="utf-8")
    target = IdeaNote.from_path(target_path, root=tmp_path)

    backlinks = IdeaGraph(IdeaSpace(tmp_path)).get_backlinks(target)

    assert len(backlinks) == 1
    assert backlinks[0].source_path == source_path
