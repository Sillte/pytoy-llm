import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from urllib.parse import unquote

from pytoy_llm.idea.domain.links import (
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
from pytoy_llm.idea.domain.uri import Uri


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
        if target.scheme == "repo":
            if not target.authority:
                return UnresolvedLink(
                    link_source=link_source, source_path=file_path, reason="Invalid repo URI"
                )
            base_path = _get_repo_folder(file_path)

            return ResolvedLocalLink.from_any(
                file_path=base_path / target.authority / target.path.lstrip("/"),
                location=location_from_fragment(link_source.fragment),
                link_source=link_source,
                source_path=file_path,
                root=self.root,
            )
        elif target.scheme in {"http", "https"}:
            return ResolvedRemoteLink.from_any(
                url=str(target), link_source=link_source, source_path=file_path
            )
        elif not target.scheme:
            path = (file_path.parent / target.path).resolve()
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
            reason=f"Unknown `scheme={target.scheme=}`. {target=}",
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

        if not target.path and not target.authority:
            return UnresolvedLink(
                link_source=link_source,
                source_path=file_path,
                reason="Empty WikiLink target",
            )

        relative_path = Path(target.path or target.authority)

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


class UriLocalPathResolver:
    def __init__(self, scheme_to_folder_path: Mapping[str, str | Path]):
        paths: dict[str, Path] = {}
        for scheme, folder_path in scheme_to_folder_path.items():
            normalized_scheme = scheme.lower()
            path = Path(folder_path)
            if not path.is_absolute():
                raise ValueError(f"Relative path is not accepted. {scheme=}, {path=}")
            if normalized_scheme in paths:
                raise ValueError(f"Duplicate URI scheme: {scheme}")
            paths[normalized_scheme] = path.resolve()
        self._scheme_to_folder_path = MappingProxyType(paths)

    @property
    def scheme_to_folder_path(self) -> Mapping[str, Path]:
        return self._scheme_to_folder_path

    def is_target_scheme(self, scheme: str) -> bool:
        return scheme.lower() in self._scheme_to_folder_path

    def resolve(self, uri: Uri, source_base_directory: str | Path | None = None) -> Path:
        """Return the absolute path in the file system."""
        path = unquote(uri.path)
        scheme = uri.scheme.lower()

        if scheme:
            if scheme not in self._scheme_to_folder_path:
                raise ValueError(
                    f"The scheme `{uri.scheme=}` is not registered, {self.scheme_to_folder_path=}"
                )
            if uri.authority:
                raise ValueError(f"URI authority is not supported for {uri.scheme=}: {uri}")

            root = self._scheme_to_folder_path[scheme]
            target_path = (root / path.lstrip("/")).resolve()
            if not target_path.is_relative_to(root):
                raise ValueError(f"URI target is outside its registered root: {uri}")
            return target_path

        if uri.authority:
            raise ValueError(f"URI authority is not supported without a scheme: {uri}")
        if source_base_directory is None:
            raise ValueError("source_base_directory is required for a URI without a scheme.")

        base_directory = Path(source_base_directory)
        if not base_directory.is_absolute():
            raise ValueError("source_base_directory must be an absolute path.")
        relative_path = Path(path)
        if relative_path.is_absolute():
            raise ValueError(f"uri.path must be relative when the URI has no scheme: {uri}")
        return (base_directory / relative_path).resolve()
