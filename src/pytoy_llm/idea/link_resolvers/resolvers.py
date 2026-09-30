from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from urllib.parse import quote, unquote

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
    def __init__(
        self,
        local_path_resolver: UriLocalPathResolver,
        *,
        default_boundary_directory: Path | None = None,
    ):
        self.local_path_resolver = local_path_resolver
        self.boundary_root_directory = (
            Path(default_boundary_directory).resolve() if default_boundary_directory else None
        )

    def resolve(
        self, file_path: Path, link_source: MarkdownLinkSource
    ) -> ResolvedLink | UnresolvedLink:
        target = link_source.target

        if self.local_path_resolver.is_registered(
            link_source.target.scheme, link_source.target.authority
        ):
            return ResolvedLocalLink.from_any(
                target_path=self.local_path_resolver.resolve(target),
                target_location=location_from_fragment(link_source.fragment),
                link_source=link_source,
                source_path=file_path,
            )

        elif target.scheme != "":
            return ResolvedRemoteLink.from_any(
                uri=target, link_source=link_source, source_path=file_path
            )
        else:
            target_path = self.local_path_resolver.resolve(
                target, source_base_directory=file_path.parent
            )
            if self.boundary_root_directory is not None:
                if not target_path.is_relative_to(self.boundary_root_directory):
                    return UnresolvedLink(
                        link_source=link_source,
                        source_path=file_path,
                        reason=f"Out of boundary: {self.boundary_root_directory=}, {target=}",
                    )
            return ResolvedLocalLink.from_any(
                target_path=target_path,
                target_location=location_from_fragment(link_source.fragment),
                link_source=link_source,
                source_path=file_path,
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
    """

    Note: SchemeDirectory.root_folder has been `resolved`.
    """

    def __init__(
        self,
        scheme_directories: Iterable[SchemeDirectory],
    ):
        directories: dict[tuple[str, str], SchemeDirectory] = {}
        for directory in scheme_directories:
            if directory.scheme == "":
                raise ValueError(f"Empty scheme is not accepted. `{directory=}`.")
            key = (directory.scheme.lower(), directory.authority)
            if key in directories:
                raise ValueError(f"Duplicate URI route: {key}")

            directories[key] = directory
        self._directories = directories

    @classmethod
    def from_any(
        cls,
        scheme_directories: Iterable[SchemeDirectory],
    ) -> Self:
        return cls(scheme_directories=scheme_directories)

    def get_root_directory(self, scheme: str, authority: str = "") -> Path | None:
        result = self._directories.get((scheme.lower(), authority))
        if result is not None:
            return result.root_directory
        return None

    def is_registered(self, scheme: str, authority: str = "") -> bool:
        return (scheme.lower(), authority) in self._directories

    def resolve(self, uri: Uri, source_base_directory: str | Path | None = None) -> Path:
        """Return the absolute path in the file system.

        Raise:
            ValueError:
                If `Uri` cannot be converted to `local_path`.(Physical path in the file system.)
            PermissionError:
                If `Uri` is outside of the root directory.
        """
        path = unquote(uri.path)
        scheme = uri.scheme.lower()

        if scheme:
            route = self._directories.get((scheme.lower(), uri.authority))
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

        return target

    def to_uri(self, local_path: Path | str, scheme: str, authority: str) -> Uri:
        scheme = scheme.lower()
        route = self._directories.get((scheme, authority))
        local_path = Path(local_path)

        if route is None:
            raise ValueError(f"URI route is not registered: {scheme=}, {authority=}.")
        if not local_path.is_absolute():
            raise ValueError(f"Local path must be absolute: {local_path=}.")

        local_path = local_path.resolve()
        posix = local_path.relative_to(route.root_directory).as_posix()
        path = "" if posix == "." else quote(posix, safe="/")

        return Uri(
            scheme=scheme,
            authority=authority,
            path=path,
        )
