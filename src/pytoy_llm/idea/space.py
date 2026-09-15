from __future__ import annotations

from pathlib import Path
from typing import Callable, Final, Self, Sequence

from .domain.exceptions import OutsidePathError
from .domain.readers import DiskFileReader, FileReaderProtocol
from .note import IdeaNote


def default_note_predicator(file_path: Path) -> bool:
    return file_path.suffix == ".md"


def _to_path_folder(path: Path):
    return path if path.is_dir() else path.parent


class IdeaSpace:
    CONVENTION_CANDIDATES: Final[tuple[str, ...]] = (".convention.md", ".index.md", "index.md")
    SPACE_META_NAME: Final[str] = ".space_meta"
    ROOT_FILE_NAME: Final[str] = ".idea_space_root"

    def __init__(
        self,
        path: Path | str,
        *,
        root: str | Path | None = None,
        file_reader: FileReaderProtocol | None = None,
        note_predicator: Callable[[Path], bool] | None = None,
        ensure_root_marker: bool = False,
    ) -> None:
        path = Path(path)
        file_reader = file_reader or DiskFileReader()
        note_predicator = note_predicator or default_note_predicator

        if root is None:
            path = path.resolve()
            root = root or self.find_idea_space_root_path(path)
        else:
            root = Path(root).resolve()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()

        if not path.is_relative_to(root):
            raise OutsidePathError(f"Space must be inside root: path={path}, root={root}")

        self._root = Path(root)
        self._relative_path = path.relative_to(root)
        self._file_reader = file_reader
        self._note_predicator = note_predicator
        if ensure_root_marker:
            self.ensure_root_marker()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        root: str | Path | None = None,
        *,
        file_reader: FileReaderProtocol | None = None,
        note_predicator: Callable[[Path], bool] | None = None,
        ensure_marker: bool = False,
    ) -> Self:

        path = Path(path)
        return cls(
            path=path,
            root=root,
            file_reader=file_reader,
            note_predicator=note_predicator,
            ensure_root_marker=ensure_marker,
        )

    def resolve(self, path: str | Path) -> Path:
        """Return the absolute path (file_path).

        Raises `OutsidePathError` if the given path is outside of `IdeaSpace`.
        """
        path = Path(path)
        if path.is_absolute():
            path = path.resolve()
        else:
            path = (self._root / path).resolve()

        if not path.is_relative_to(self._root):
            raise OutsidePathError(f"`{path}` is outside of `IdeaSpace`.")
        return path

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

    @property
    def convention(self) -> IdeaNote | None:
        for cand in self.CONVENTION_CANDIDATES:
            if (self.folder_path / cand).exists():
                return IdeaNote.from_path(
                    file_path=self.folder_path / cand, root=self.root, file_reader=self._file_reader
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
                    if not self._is_meta_folder(child):
                        visit(child, current_depth + 1)

        visit(self.folder_path, 0)

        return sorted(tuple(notes), key=lambda note: note.path)

    def _is_note_path(self, file_path: Path) -> bool:
        return self._note_predicator(file_path)

    def _is_meta_folder(self, folder_path: Path) -> bool:
        return folder_path.name == self.SPACE_META_NAME

    def ensure_root_marker(self) -> None:
        (self.root / self.ROOT_FILE_NAME).touch()

    @classmethod
    def find_idea_space_root_path(cls, start_path: Path | str) -> Path:
        """Find the IdeaSpace root associated with the given path.

        Searches the given path and its ancestors for an IdeaSpace root marker.
        If no marker is found, the folder containing `start_path` is returned.
        """
        start_folder = _to_path_folder(Path(start_path))

        current = start_folder
        while True:
            if (current / cls.ROOT_FILE_NAME).exists():
                return current

            parent = current.parent
            if parent == current:
                return start_folder
            current = parent
