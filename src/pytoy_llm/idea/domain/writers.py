from pathlib import Path
from typing import Protocol


class FileWriterProtocol(Protocol):
    """Write the text.

    Typically, it is intended to be write the file content in disk.
    Sometimes, an editor may write the text to the buffer not disk.
    This protocol is defined to support this situation.
    """

    def write(self, text: str, file_path: Path) -> None: ...


class DiskFileWriter:
    def __init__(self, parents: bool = True):
        self._parents = parents

    def write(self, text: str, file_path: Path):
        if self._parents:
            file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text(text, encoding="utf8")
