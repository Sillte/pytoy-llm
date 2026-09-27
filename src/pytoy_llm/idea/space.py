from __future__ import annotations

from pathlib import Path
from typing import Final, Self, Sequence

from pytoy_llm.idea.domain.path import (
    ROOT_MARKER_FILE_NAME,
    IdeaPath,
    resolve_absolute_path,
    resolve_root_folder_path,
)

from .domain.exceptions import OutsidePathError
from .domain.readers import DiskFileReader, FileReaderProtocol
from .note import IdeaNote


def note_predicator_v1(file_path: Path) -> bool:
    return file_path.suffix == ".md"


class IdeaSpace:
    ROOT_MARKER_FILE_NAME: Final[str] = ROOT_MARKER_FILE_NAME
    CONVENTION_CANDIDATES: Final[tuple[str, ...]] = (".convention.md", ".idea_space_convention.md")
    SPACE_META_NAME: Final[str] = ".space_meta"

    def __init__(
        self,
        path: Path | str,
        *,
        root: str | Path | None = None,
        file_reader: FileReaderProtocol | None = None,
        ensure_root_marker: bool = False,
    ) -> None:

        root_folder_path = resolve_root_folder_path(root, path)
        absolute_path = resolve_absolute_path(path, root_folder_path)
        file_reader = file_reader or DiskFileReader()

        if not absolute_path.is_relative_to(root_folder_path):
            raise OutsidePathError(f"Space must be inside root: path={path}, root={root}")
        if absolute_path.exists() and not absolute_path.is_dir():
            raise ValueError(f"Space path must be a folder, not a: `{path=}`, `{absolute_path=}`")

        self._root_folder_path = root_folder_path
        self._relative_path = absolute_path.relative_to(self._root_folder_path)
        self._file_reader = file_reader
        self._note_predicator = note_predicator_v1

        if ensure_root_marker:
            self.ensure_root_marker()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        root: str | Path | None = None,
        *,
        file_reader: FileReaderProtocol | None = None,
        with_creation: bool = False,
        ensure_root_marker: bool = False,
    ) -> Self:
        path = Path(path)

        if with_creation:
            path.mkdir(exist_ok=True, parents=True)

        return cls(
            path=path,
            root=root,
            file_reader=file_reader,
            ensure_root_marker=ensure_root_marker,
        )

    @classmethod
    def from_note(
        cls,
        note_or_path: IdeaNote | str | Path,
        *,
        ensure_root_marker: bool = False,
    ) -> Self:
        if not isinstance(note_or_path, IdeaNote):
            note_or_path = IdeaNote.from_path(note_or_path)
        return cls.from_path(
            path=note_or_path.file_path.parent,
            root=note_or_path.root_folder_path,
            file_reader=note_or_path.file_reader,
            ensure_root_marker=ensure_root_marker,
        )

    def resolve(self, path: str | Path) -> Path:
        """Return the absolute path (file_path).

        Raises `OutsidePathError` if the given path is outside of `IdeaSpace`.
        """
        absolute_path = resolve_absolute_path(path, self._root_folder_path)

        if not absolute_path.is_relative_to(self._root_folder_path):
            raise OutsidePathError(f"`{absolute_path}` is outside of `IdeaSpace`.")
        return absolute_path

    @property
    def idea_path(self) -> IdeaPath:
        return self._relative_path.as_posix()

    @property
    def path(self) -> IdeaPath:
        return self.idea_path

    @property
    def folder_path(self) -> Path:
        return self._root_folder_path / self._relative_path

    @property
    def root_folder_path(self) -> Path:
        return self._root_folder_path

    @property
    def root_space(self) -> Self:
        return self.from_path(
            path=self.root_folder_path,
            root=self.root_folder_path,
            file_reader=self._file_reader,
        )

    @property
    def parent(self) -> Self:
        return self.from_path(
            path=self._relative_path.parent,
            root=self._root_folder_path,
            file_reader=self._file_reader,
        )

    @property
    def relative_parts(self) -> Sequence[str]:
        return self._relative_path.parts

    @property
    def convention(self) -> IdeaNote | None:
        for cand in self.CONVENTION_CANDIDATES:
            if (self.folder_path / cand).exists():
                return IdeaNote.from_path(
                    path=self.folder_path / cand,
                    root=self.root_folder_path,
                    file_reader=self._file_reader,
                )
        return None

    @property
    def space_meta_folder(self) -> Path:
        return self.folder_path / self.SPACE_META_NAME

    def get_subspaces(self, depth: int | None = 0) -> Sequence["IdeaSpace"]:
        spaces: list[IdeaSpace] = []

        def visit(path: Path, current_depth: int) -> None:
            if depth is not None and depth < current_depth:
                return

            for child in path.iterdir():
                if child.is_dir():
                    if self._is_meta_folder(child):
                        continue
                    spaces.append(
                        self.from_path(
                            path=child,
                            root=self.root_folder_path,
                            file_reader=self._file_reader,
                        )
                    )
                    visit(child, current_depth + 1)

        visit(self.folder_path, 0)

        return sorted(tuple(spaces), key=lambda space: space.idea_path)

    def get_notes(
        self,
        depth: int | None = 0,
    ) -> Sequence[IdeaNote]:
        notes: list[IdeaNote] = []

        def visit(path: Path, current_depth: int) -> None:

            if depth is not None and depth < current_depth:
                return

            for child in path.iterdir():
                if child.is_file():
                    if self._is_note_path(child):
                        notes.append(
                            IdeaNote.from_path(
                                child, root=self.root_folder_path, file_reader=self._file_reader
                            )
                        )
                elif child.is_dir():
                    if not self._is_meta_folder(child):
                        visit(child, current_depth + 1)

        visit(self.folder_path, 0)

        return sorted(tuple(notes), key=lambda note: note.idea_path)

    def _is_note_path(self, file_path: Path) -> bool:
        return self._note_predicator(file_path)

    def _is_meta_folder(self, folder_path: Path) -> bool:
        return folder_path.name == self.SPACE_META_NAME

    def ensure_root_marker(self) -> None:
        (self.root_folder_path / self.ROOT_MARKER_FILE_NAME).touch()
