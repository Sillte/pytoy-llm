import subprocess
from pathlib import Path
from urllib.parse import urlparse

from ..domain.links import (
    AnchorLocation,
    LineLocation,
    LinkSource,
    Location,
    MarkdownLinkSource,
    ResolvedLink,
    ResolvedLocalLink,
    ResolvedRemoteLink,
    UnresolvedLink,
    WikiLinkSource,
)


def location_from_fragment(
    fragment: str | None,
) -> Location:
    """Convert a link fragment to a target location.

    ``#Lx`` is converted to a zero-based line location. ``#Lx-Ly`` is
    accepted for compatibility, but currently retains only its start line;
    range targets will be represented by a separate location type later.
    """
    if fragment is None:
        return LineLocation()

    if fragment.startswith("L"):
        if fragment[1:].isdigit():
            return LineLocation(
                line=int(fragment[1:]) - 1,
            )
        hyphen_position = fragment.find("-")
        if hyphen_position != -1:
            start = fragment[1:hyphen_position]
            end = fragment[hyphen_position + 1 :]

            if start.isdigit() and end.isdigit():
                return LineLocation(
                    line=int(start) - 1,
                )
    return AnchorLocation(fragment)


def _get_repo_folder(file_path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", file_path.parent.as_posix(), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return Path(result.stdout.strip())


class MarkdownLinkResolver:
    def __init__(self, root: Path):
        self.root = root

    def resolve(
        self, file_path: Path, link_source: MarkdownLinkSource
    ) -> ResolvedLink | UnresolvedLink:
        target = link_source.target
        parsed = urlparse(target)
        if parsed.scheme == "repo":
            if not target.startswith("repo://"):
                return UnresolvedLink(
                    link_source=link_source, source_path=file_path, reason="Invalid repo URI"
                )
            base_path = _get_repo_folder(file_path)
            _, _, target = target.partition("repo://")

            return ResolvedLocalLink.from_any(
                file_path=base_path / target,
                location=location_from_fragment(link_source.fragment),
                link_source=link_source,
                source_path=file_path,
                root=self.root,
            )
        elif parsed.scheme in {"http", "https"}:
            return ResolvedRemoteLink.from_any(
                url=target, link_source=link_source, source_path=file_path
            )
        elif not parsed.scheme:
            path = (file_path.parent / target).resolve()
            return ResolvedLocalLink.from_any(
                file_path=path,
                location=location_from_fragment(link_source.fragment),
                link_source=link_source,
                source_path=file_path,
                root=self.root,
            )

        return UnresolvedLink(
            link_source=link_source,
            source_path=file_path,
            reason=f"Unknown `scheme={parsed.scheme=}`. {parsed=}",
        )


class WikiLinkResolver:
    """
    TODO: Correspondence of `target` and the actual path should be revised later.
    """

    def __init__(self, root: Path):
        self.root = root

    def resolve(
        self,
        file_path: Path,
        link_source: WikiLinkSource,
    ) -> ResolvedLink | UnresolvedLink:
        target = link_source.target

        if not target:
            return UnresolvedLink(
                link_source=link_source,
                source_path=file_path,
                reason="Empty WikiLink target",
            )

        relative_path = Path(target)

        if relative_path.suffix == "":
            relative_path = relative_path.with_suffix(".md")

        return ResolvedLocalLink.from_any(
            file_path=self.root / relative_path,
            location=location_from_fragment(link_source.fragment),
            link_source=link_source,
            source_path=file_path,
            root=self.root,
        )


class LinkSourceResolver:
    def __init__(self, root: Path):
        self.root = root
        self.markdown_resolver = MarkdownLinkResolver(root)
        self.wiki_resolver = WikiLinkResolver(root)

    def resolve(
        self,
        file_path: Path,
        link_source: LinkSource,
    ) -> ResolvedLink | UnresolvedLink:
        match link_source:
            case WikiLinkSource():
                return self.wiki_resolver.resolve(file_path, link_source)
            case MarkdownLinkSource():
                return self.markdown_resolver.resolve(
                    file_path,
                    link_source,
                )
            case _:
                return UnresolvedLink(link_source=link_source, source_path=file_path)
