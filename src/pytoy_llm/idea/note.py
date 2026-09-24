from dataclasses import dataclass
from pathlib import Path
from typing import Self, Sequence

from pytoy_llm.idea.domain.links import LinkSource, LinkSourceExtractor
from pytoy_llm.idea.domain.metadata import MetaDataProtocol
from pytoy_llm.idea.domain.readers import (
    DiskFileReader,
    FileReaderProtocol,
)
from pytoy_llm.idea.domain.writers import DiskFileWriter, FileWriterProtocol
from pytoy_llm.idea.infra.yamlrock_wrapper import YamlRockWrapper
from pytoy_llm.idea.link_extractors.naive_source_extractors import (
    NaiveLinkSourceExtractor,
)


@dataclass(frozen=True)
class Interpretation:
    metadata: MetaDataProtocol | None
    body: str
    body_start_line: int


def _is_frontmatter_delimiter(line: str) -> bool:
    return line.strip(" \t\r\n") == "---"


def interpret(text: str) -> Interpretation:
    lines = text.splitlines(keepends=True)
    # --------------------------------------------------------
    # Remove leading blank lines.
    # --------------------------------------------------------

    start = 0

    while start < len(lines):
        line = lines[start]

        if line.strip(" \t\r\n") != "":
            break

        start += 1

    # Entire document is blank.
    if start == len(lines):
        return Interpretation(metadata=None, body=text, body_start_line=0)

    # --------------------------------------------------------
    # No front matter.
    # --------------------------------------------------------

    if not _is_frontmatter_delimiter(lines[start]):
        return Interpretation(metadata=None, body=text, body_start_line=0)

    # --------------------------------------------------------
    # Find closing front-matter delimiter.
    # --------------------------------------------------------

    for i in range(start + 1, len(lines)):
        line = lines[i]

        if not _is_frontmatter_delimiter(line):
            continue

        yaml_text = "".join(lines[start + 1 : i])

        body_start = i + 1

        body = "".join(lines[body_start:])

        metadata = YamlRockWrapper.from_yaml_text(yaml_text)

        return Interpretation(metadata=metadata, body=body, body_start_line=body_start)

    # --------------------------------------------------------
    # No closing delimiter.
    #
    # Treat the entire document as Markdown.
    # --------------------------------------------------------
    return Interpretation(metadata=None, body=text, body_start_line=0)


def _to_text(metadata: MetaDataProtocol, body: str) -> str:
    """
    Serialize the IdeaNote.
    The output is normalized as:
        ---
        YAML
        ---
        Markdown body
    """

    if len(metadata) == 0:
        return body

    yaml_text = metadata.as_text().rstrip("\r\n")
    return f"---\n{yaml_text}\n---\n{body}"


class IdeaNote:
    """
    A single note represented by:

        YAML Front Matter
        +
        Markdown body

    Example:

        ---
        id: concepts.event-driven
        type: concept
        status: active
        tags:
          - architecture
          - ai
        ---

        # Event-driven architecture

        Some text.

    This is primarily source-mapping information.

    When `__init__` is directly invoked, it is intended that the `text` is being edited, that is,
    the document of `path` is being modified with the text editor and the intended text is different from the file content in disk.
    """

    def __init__(
        self,
        text: str,
        path: Path,
        root: Path,
        *,
        link_source_extractor: LinkSourceExtractor | None = None,
    ) -> None:
        path = Path(path)
        root = root.resolve()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        link_source_extractor = link_source_extractor or NaiveLinkSourceExtractor()

        if not path.is_relative_to(root):
            raise PermissionError(f"Path must be inside root: path={path}, root={root}")
        self._relative_path = path.relative_to(root)
        self._root = root
        interpreted_result = interpret(text)
        self._metadata = interpreted_result.metadata or YamlRockWrapper()
        self._body = interpreted_result.body
        self._body_start_line = interpreted_result.body_start_line
        self._link_source_extractor = link_source_extractor

    @classmethod
    def from_path(
        cls,
        file_path: str | Path,
        root: Path | None = None,
        *,
        link_source_extractor: LinkSourceExtractor | None = None,
        file_reader: FileReaderProtocol | None = None,
    ) -> Self:
        """
        Parse an IdeaNote from the original document.
        """
        file_path = Path(file_path)
        file_reader = file_reader or DiskFileReader()
        text = file_reader.read(file_path)
        root = root or file_path.parent
        return cls(
            text=text, path=file_path, root=root, link_source_extractor=link_source_extractor
        )

    @classmethod
    def create(
        cls,
        file_path: Path,
        body: str,
        metadata: MetaDataProtocol | None = None,
        root: Path | None = None,
        *,
        link_source_extractor: LinkSourceExtractor | None = None,
    ) -> Self:
        metadata = metadata or YamlRockWrapper()
        text = _to_text(metadata, body)
        root = root or file_path.parent
        return cls(
            text=text, path=file_path, root=root, link_source_extractor=link_source_extractor
        )

    @property
    def path(self) -> str:
        return self._relative_path.as_posix()

    @property
    def file_path(self) -> Path:
        return self._root / self._relative_path

    @property
    def root(self) -> Path:
        return self._root

    @property
    def text(self) -> str:
        return self.to_text()

    @property
    def body(self) -> str:
        return self._body

    @property
    def body_start_line(self) -> int:
        """0-based line where the Markdown body starts in the original text."""
        return self._body_start_line

    @property
    def metadata(self) -> MetaDataProtocol:
        return self._metadata

    def set_body(self, body: str) -> None:
        self._body = body

    def to_text(self) -> str:
        """
        Serialize the IdeaNote.
        The output is normalized as:
            ---
            YAML
            ---
            Markdown body
        """
        return _to_text(self._metadata, self._body)

    def write(self, file_writer: FileWriterProtocol | None = None) -> None:
        """Write the serialized note to :attr:`file_path`."""
        file_writer = file_writer or DiskFileWriter()
        file_writer.write(self.to_text(), self.file_path)

    @property
    def link_sources(self) -> Sequence[LinkSource]:
        """Source links contained in the original text."""

        return tuple(self._link_source_extractor.extract(self.text))
