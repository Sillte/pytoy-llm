from __future__ import annotations

from pathlib import Path
from typing import Callable, Self, Sequence

from .domain.readers import DiskFileReader, FileReaderProtocol
from .note import IdeaNote


def default_note_predicator(file_path: Path) -> bool:
    return file_path.suffix == ".md"


class IdeaSpace:
    def __init__(
        self,
        path: Path | str,
        *,
        root: Path | None = None,
        file_reader: FileReaderProtocol | None = None,
        note_predicator: Callable[[Path], bool] | None = None,
    ) -> None:
        path = Path(path)
        file_reader = file_reader or DiskFileReader()
        note_predicator = note_predicator or default_note_predicator

        if root is None:
            path = path.resolve()
            root = path
        else:
            root = root.resolve()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()

        if not path.is_relative_to(root):
            raise ValueError(f"Space must be inside root: path={path}, root={root}")

        self._root = root
        self._relative_path = path.relative_to(root)
        self._file_reader = file_reader
        self._note_predicator = note_predicator

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        root: str | Path | None = None,
        *,
        file_reader: FileReaderProtocol | None = None,
        note_predicator: Callable[[Path], bool] | None = None,
    ) -> Self:
        def _to_path_folder(path: Path):
            return path if path.is_dir() else path.parent

        path = Path(path)
        root = root or _to_path_folder(path)
        return cls(
            path=path, root=Path(root), file_reader=file_reader, note_predicator=note_predicator
        )

    @property
    def path(self) -> str:
        return self._relative_path.as_posix()

    @property
    def folder_path(self) -> Path:
        return self._root / self._relative_path

    @property
    def root(self) -> Path:
        return self._root

    @property
    def parent(self) -> "IdeaSpace":
        return IdeaSpace(
            path=self._relative_path.parent,
            root=self._root,
            file_reader=self._file_reader,
            note_predicator=self._note_predicator,
        )

    @property
    def relative_parts(self) -> Sequence[str]:
        return self._relative_path.parts

    def get_subspaces(self, depth: int | None = 0) -> Sequence["IdeaSpace"]:
        spaces: list[IdeaSpace] = []

        def visit(path: Path, current_depth: int) -> None:
            if depth is not None and depth < current_depth:
                return

            for child in path.iterdir():
                if child.is_dir():
                    spaces.append(
                        IdeaSpace(
                            path=child,
                            root=self.root,
                            file_reader=self._file_reader,
                            note_predicator=self._note_predicator,
                        )
                    )
                    visit(child, current_depth + 1)

        visit(self.folder_path, 0)

        return sorted(tuple(spaces), key=lambda space: space.path)

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
                            IdeaNote.from_path(child, root=self.root, file_reader=self._file_reader)
                        )
                elif child.is_dir():
                    visit(child, current_depth + 1)

        visit(self.folder_path, 0)

        return sorted(tuple(notes), key=lambda note: note.path)

    def _is_note_path(self, file_path: Path) -> bool:
        return self._note_predicator(file_path)
