from pathlib import Path
from typing import Self, Sequence, overload

from .domain.links import (
    IdeaLink,
    UnresolvedIdeaLink,
    UnresolvedLink,
)
from .domain.readers import FileReaderProtocol
from .domain.uri import Uri
from .link_converter import IdeaLinkConverter
from .link_resolvers import MarkdownLinkResolver, SchemeDirectory, UriLocalPathResolver
from .note import IdeaNote
from .space import IdeaSpace


class IdeaGraph:
    def __init__(self, space: IdeaSpace) -> None:
        self._space = space
        scheme_directories = [
            SchemeDirectory.from_any(root_directory=self._space.root_directory_path, scheme="idea")
        ]
        self._path_resolver = UriLocalPathResolver(
            scheme_directories, default_root_directory=self._space.root_directory_path
        )
        self._link_resolver = MarkdownLinkResolver(self._path_resolver)
        self._converter = IdeaLinkConverter()

    @classmethod
    def from_path(
        cls,
        root_path: Path,
        *,
        file_reader: FileReaderProtocol | None = None,
    ) -> Self:
        return cls(
            space=IdeaSpace.from_path(path=root_path, root=root_path, file_reader=file_reader)
        )

    @classmethod
    def from_space(
        cls,
        space: IdeaSpace,
    ) -> Self:
        return cls(space=space)

    @classmethod
    def from_note(
        cls,
        note: IdeaNote,
        *,
        file_reader: FileReaderProtocol | None = None,
    ) -> Self:
        return cls.from_space(
            space=IdeaSpace.from_path(
                path=note.root_directory_path,
                root=note.root_directory_path,
                file_reader=file_reader,
            )
        )

    @property
    def space(self) -> IdeaSpace:
        return self._space

    @overload
    def resolve_links(
        self, source_note: IdeaNote, *, only_valid: bool = True
    ) -> Sequence[IdeaLink]: ...
    @overload
    def resolve_links(
        self, source_note: IdeaNote, *, only_valid: bool = False
    ) -> Sequence[IdeaLink | UnresolvedIdeaLink]: ...
    def resolve_links(
        self, source_note: IdeaNote, *, only_valid: bool = True
    ) -> Sequence[IdeaLink | UnresolvedIdeaLink]:
        """Resolve links contained in ``source_note``.

        Links that cannot be resolved because their targets are invalid or
        inaccessible are returned as ``UnresolvedIdeaLink`` values. They do
        not raise an exception.
        """
        resolver = self._link_resolver
        inner_links = []
        for link_source in source_note.link_sources:
            try:
                link = resolver.resolve(source_note.file_path, link_source)
            except (ValueError, OSError) as exc:
                link = UnresolvedLink(
                    link_source=link_source, source_path=source_note.file_path, reason=str(exc)
                )
            inner_links.append(link)
        links = [self._converter.convert(link) for link in inner_links]
        if only_valid:
            return [link for link in links if isinstance(link, IdeaLink)]
        else:
            return links

    def get_backlinks(self, destination_note: IdeaNote) -> Sequence[IdeaLink]:
        result = []
        destination_uri = Uri.from_any(destination_note.file_path.as_uri())
        for note in self._space.get_notes(depth=None):
            links = self.resolve_links(note, only_valid=True)
            result += [link for link in links if link.uri == destination_uri]

        return result
