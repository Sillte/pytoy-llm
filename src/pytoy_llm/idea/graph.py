from pathlib import Path
from typing import Callable, Self, Sequence, overload

from .domain.links import (
    IdeaLink,
    UnresolvedIdeaLink,
    UnresolvedLink,
)
from .domain.readers import FileReaderProtocol
from .link_converter import IdeaLinkConverter
from .link_resolvers import LinkSourceResolver
from .note import IdeaNote
from .space import IdeaSpace


class IdeaGraph:
    def __init__(self, space: IdeaSpace) -> None:
        self._space = space
        self._resolver = LinkSourceResolver(self._space.root)
        self._converter = IdeaLinkConverter()

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        file_reader: FileReaderProtocol | None = None,
        note_predicator: Callable[[Path], bool] | None = None,
    ) -> Self:
        return cls(
            space=IdeaSpace.from_path(
                path=root, root=root, file_reader=file_reader, note_predicator=note_predicator
            )
        )

    @classmethod
    def from_note(
        cls,
        note: IdeaNote,
        *,
        file_reader: FileReaderProtocol | None = None,
        note_predicator: Callable[[Path], bool] | None = None,
    ) -> Self:
        return cls(
            space=IdeaSpace.from_path(
                path=note.root,
                root=note.root,
                file_reader=file_reader,
                note_predicator=note_predicator,
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
        resolver = self._resolver
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
        for note in self._space.get_notes(depth=None):
            links = self.resolve_links(note, only_valid=True)
            result += [
                link for link in links if link.target_file_path == destination_note.file_path
            ]

        return result
