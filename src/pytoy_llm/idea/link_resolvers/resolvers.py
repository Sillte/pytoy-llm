from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from urllib.parse import unquote

from pytoy_llm.idea.domain.exceptions import OutsidePathError
from pytoy_llm.idea.domain.links import (
    AnchorLocation,
    LineLocation,
    Location,
    MarkdownLinkSource,
    ResolvedLink,
    ResolvedLocalLink,
    ResolvedRemoteLink,
    UnresolvedLink,
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


class MarkdownLinkResolver:
    def __init__(self, local_path_resolver: UriLocalPathResolver):
        self.local_path_resolver = local_path_resolver

    def resolve(
        self, file_path: Path, link_source: MarkdownLinkSource
    ) -> ResolvedLink | UnresolvedLink:
        target = link_source.target

        if target.scheme in {"http", "https"}:
            return ResolvedRemoteLink.from_any(
                url=str(target), link_source=link_source, source_path=file_path
            )
        elif not target.scheme:
            return ResolvedLocalLink.from_any(
                target_path=self.local_path_resolver.resolve(
                    target, source_base_directory=file_path.parent
                ),
                target_location=location_from_fragment(link_source.fragment),
                link_source=link_source,
                source_path=file_path,
            )

        return UnresolvedLink(
            link_source=link_source,
            source_path=file_path,
            reason=f"Unknown `scheme={target.scheme=}`. {target=}",
        )


@dataclass(frozen=True)
class SchemeDirectory:
    root_directory: Path
    scheme: str
    authority: str = ""

    def __post_init__(self) -> None:
        if not self.scheme:
            raise ValueError("scheme must not be empty.")
        if not self.root_directory.is_absolute():
            raise ValueError(f"root_directory must be absolute: {self.root_directory}")
        object.__setattr__(self, "scheme", self.scheme.lower())
        object.__setattr__(self, "root_directory", self.root_directory.resolve())

    @classmethod
    def from_any(cls, root_directory: str | Path, scheme: str, authority: str = "") -> Self:
        return cls(Path(root_directory).resolve(), scheme=scheme, authority=authority)


class UriLocalPathResolver:
    def __init__(
        self,
        scheme_directories: Iterable[SchemeDirectory],
        default_root_directory: Path | None = None,
    ):
        directories: dict[tuple[str, str], SchemeDirectory] = {}
        for directory in scheme_directories:
            key = (directory.scheme, directory.authority)
            if key in directories:
                raise ValueError(f"Duplicate URI route: {key}")
            directories[key] = directory
        self._directories = directories
        self._default_root_directory = (
            Path(default_root_directory).resolve() if default_root_directory else None
        )

    def is_registered(self, scheme: str, authority: str = "") -> bool:
        return (scheme.lower(), authority) in self._directories

    def resolve(self, uri: Uri, source_base_directory: str | Path | None = None) -> Path:
        """Return the absolute path in the file system.

        Raise:
            ValueError:
                If `Uri` cannot be converted to `local_path`.(Physical path in the file system.)
            PermissionError:
                If `Uri` is outside of `root_folder`.
        """
        path = unquote(uri.path)
        scheme = uri.scheme.lower()

        if scheme:
            route = self._directories.get((scheme, uri.authority))
            if route is None:
                raise ValueError(f"URI route is not registered: {scheme=}, {uri.authority=}.")

            root = route.root_directory
            target_path = (root / path.lstrip("/")).resolve()
            if not target_path.is_relative_to(root):
                raise OutsidePathError(f"URI target is outside its registered root: {uri}")
            return target_path

        if uri.authority:
            raise ValueError(f"URI authority is not supported without a scheme: {uri}")
        if source_base_directory is None:
            raise ValueError("`source_base_directory` is required for a URI without a scheme.")

        base_directory = Path(source_base_directory)
        if not base_directory.is_absolute():
            raise ValueError("source_base_directory must be an absolute path.")
        relative_path = Path(path)
        if relative_path.is_absolute():
            raise ValueError(f"uri.path must be relative when the URI has no scheme: {uri}")
        target = (base_directory / relative_path).resolve()
        if self._default_root_directory:
            if not target.is_relative_to(self._default_root_directory):
                raise OutsidePathError(f"The given relative `{path=}` is outside its given root.")
        return target
