from pathlib import Path
from typing import Final

type IdeaPath = str  # Relative path represented by POSIX.

ROOT_MARKER_FILE_NAME: Final[str] = ".idea_space_root"


def _to_path_directory(path: Path):
    return path if path.is_dir() else path.parent


def find_root_directory_path(start_path: Path) -> Path:
    """Find the IdeaSpace root associated with the given path.

    Searches the given path and its ancestors for an IdeaSpace root marker.
    If no marker is found, the directory containing `start_path` is returned.
    """
    start_directory = _to_path_directory(Path(start_path))

    current = start_directory
    while True:
        if (current / ROOT_MARKER_FILE_NAME).exists():
            return current

        parent = current.parent
        if parent == current:
            return start_directory
        current = parent


def resolve_root_directory_path(root: str | Path | None, pivot_path: str | Path) -> Path:
    if root is None:
        return find_root_directory_path(Path(pivot_path))
    else:
        return Path(root).resolve()


def resolve_absolute_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    root_directory_path = resolve_root_directory_path(root=root, pivot_path=path)
    return (root_directory_path / path).resolve() if not path.is_absolute() else path.resolve()
