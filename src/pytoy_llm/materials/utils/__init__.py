import os
from pathlib import Path
from typing import Callable, Sequence


class FileGatherer:
    def __init__(self,
                 exclude_predicators: None | Sequence[Callable[[str], bool]] = None,
                 exclude_folders: Sequence[Path] | None = None) -> None:
        if not exclude_predicators:
            exclude_predicators = [lambda item: item.startswith("."), lambda item: item.startswith("__")]
        self._exclude_predicators = exclude_predicators
        exclude_folders = exclude_folders or []
        self._exclude_folders = set(exclude_folders)

    def gather(self, root_folder: Path | str) -> Sequence[Path]:
        root_folder = Path(root_folder)
        filepaths = []
        for root, dirs, files in os.walk(root_folder, topdown=True):
            if Path(root) in self._exclude_folders:
                dirs[:] = []
                continue
            # Logic for exclusion.
            dirs[:] = [item for item in dirs if all(not predicator(item) for predicator in self._exclude_predicators)]
            filepaths += [Path(root) / filename for filename in files]
        return filepaths
