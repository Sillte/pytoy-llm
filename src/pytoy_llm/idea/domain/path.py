from pathlib import Path
from typing import Final

type IdeaPath = str  # Relative path represented by POSIX.

ROOT_MARKER_FILE_NAME: Final[str] = ".idea_space_root"


def _to_path_folder(path: Path):
    return path if path.is_dir() else path.parent


def find_root_folder_path(start_path: Path) -> Path:
    """Find the IdeaSpace root associated with the given path.

    Searches the given path and its ancestors for an IdeaSpace root marker.
    If no marker is found, the folder containing `start_path` is returned.
    """
    start_folder = _to_path_folder(Path(start_path))

    current = start_folder
    while True:
        if (current / ROOT_MARKER_FILE_NAME).exists():
            return current

        parent = current.parent
        if parent == current:
            return start_folder
        current = parent


def resolve_root_folder_path(root: str | Path | None, pivot_path: str | Path) -> Path:
    if root is None:
        return find_root_folder_path(Path(pivot_path))
    else:
        return Path(root).resolve()


def resolve_absolute_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    root_folder_path = resolve_root_folder_path(root=root, pivot_path=path)
    return (root_folder_path / path).resolve() if not path.is_absolute() else path.resolve()
