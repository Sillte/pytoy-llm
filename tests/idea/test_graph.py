from pathlib import Path

from pytoy_llm.idea import AnchorLocation, IdeaGraph, IdeaLink, IdeaNote, IdeaSpace, LineLocation


def test_graph_resolves_markdown_and_wiki_links(tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    target_path = tmp_path / "target.md"
    other_path = tmp_path / "other.md"
    source_path.write_text("[target](target.md#L3) [[other]]", encoding="utf-8")
    target_path.write_text("target", encoding="utf-8")
    other_path.write_text("other", encoding="utf-8")
    source = IdeaNote.from_path(source_path, root=tmp_path)

    links = IdeaGraph(IdeaSpace(tmp_path)).resolve_links(source)

    assert len(links) == 2
    assert isinstance(links[0], IdeaLink)
    assert links[0].target_file_path == target_path
    assert links[0].target_location == LineLocation(line=2)
    assert isinstance(links[1].target_location, AnchorLocation) is False


def test_graph_returns_unresolved_link_for_target_outside_root(tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    source_path.write_text("[outside](../outside.md)", encoding="utf-8")
    source = IdeaNote.from_path(source_path, root=tmp_path)

    links = IdeaGraph(IdeaSpace(tmp_path)).resolve_links(source, only_valid=False)

    assert len(links) == 1
    assert links[0].__class__.__name__ == "UnresolvedIdeaLink"


def test_graph_finds_backlinks(tmp_path: Path) -> None:
    source_path = tmp_path / "source.md"
    target_path = tmp_path / "target.md"
    source_path.write_text("[target](target.md)", encoding="utf-8")
    target_path.write_text("target", encoding="utf-8")
    target = IdeaNote.from_path(target_path, root=tmp_path)

    backlinks = IdeaGraph(IdeaSpace(tmp_path)).get_backlinks(target)

    assert len(backlinks) == 1
    assert backlinks[0].source_path == source_path
