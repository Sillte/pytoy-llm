from pathlib import Path
from typing import Protocol


class FileReaderProtocol(Protocol):
    """Read the text from the filepath.

    Typically, it is intended to be read from the file content in disk.
    Sometimes, an editor may read the file in edit.
    This protocol is defined to support this situation.
    """

    def read(self, file_path: Path) -> str: ...


class DiskFileReader:
    def read(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf8")
