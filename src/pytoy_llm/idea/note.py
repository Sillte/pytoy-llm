from dataclasses import dataclass
from pathlib import Path
from typing import Self, Sequence

from pytoy_llm.idea.domain.exceptions import (
    MetadataSerializationError,
)
from pytoy_llm.idea.domain.links import LinkSource, LinkSourceExtractor
from pytoy_llm.idea.domain.metadata import MetaDataProtocol
from pytoy_llm.idea.domain.path import IdeaPath, resolve_absolute_path, resolve_root_folder_path
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

    If metadata cannot be serialized, return the body instead of
    propagating the metadata serialization error.
    """

    if len(metadata) == 0:
        return body

    try:
        yaml_text = metadata.as_text().rstrip("\r\n")
    except MetadataSerializationError:
        return body
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
        path: str | Path,
        root: Path | str | None,
        *,
        file_reader: FileReaderProtocol | None = None,
        link_source_extractor: LinkSourceExtractor | None = None,
    ) -> None:
        """Create an IdeaNote from source text.

        Raises:
            PermissionError: If ``path`` is outside ``root``.
            MetadataDeserializationError: If the source text contains invalid
                YAML front matter.
        """
        path = Path(path)
        root_folder_path = resolve_root_folder_path(root=root, pivot_path=path)
        absolute_path = resolve_absolute_path(path, root_folder_path)

        if not absolute_path.is_relative_to(root_folder_path):
            raise PermissionError(
                f"Path must be inside root: "
                f"path={path}, root_folder_path={root_folder_path}, root={root}"
            )
        self._relative_path = absolute_path.relative_to(root_folder_path)
        self._root_folder_path = root_folder_path

        interpreted_result = interpret(text)
        self._metadata = interpreted_result.metadata or YamlRockWrapper()
        self._body = interpreted_result.body
        self._body_start_line = interpreted_result.body_start_line

        self._link_source_extractor = link_source_extractor or NaiveLinkSourceExtractor()
        self._file_reader: FileReaderProtocol = file_reader or DiskFileReader()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        root: str | Path | None = None,
        *,
        link_source_extractor: LinkSourceExtractor | None = None,
        file_reader: FileReaderProtocol | None = None,
    ) -> Self:
        """Parse an IdeaNote from the original document.

        Raises:
            OSError: If the document cannot be read by ``file_reader``.
            PermissionError: If ``file_path`` is outside ``root``.
            MetadataDeserializationError: If the document contains invalid YAML
                front matter.
        """
        path = Path(path)
        file_reader = file_reader or DiskFileReader()
        text = file_reader.read(path)
        note = cls(
            text=text,
            path=path,
            root=root,
            file_reader=file_reader,
            link_source_extractor=link_source_extractor,
        )
        return note

    @classmethod
    def create(
        cls,
        file_path: str | Path,
        body: str,
        metadata: MetaDataProtocol | None = None,
        root: str | Path | None = None,
        *,
        link_source_extractor: LinkSourceExtractor | None = None,
    ) -> Self:
        """Create an IdeaNote from a body and optional metadata.

        Raises:
            PermissionError: If ``file_path`` is outside ``root``.
        """
        metadata = metadata or YamlRockWrapper()
        text = _to_text(metadata, body)
        file_path = Path(file_path)
        return cls(
            text=text, path=file_path, root=root, link_source_extractor=link_source_extractor
        )

    @property
    def idea_path(self) -> IdeaPath:
        return self._relative_path.as_posix()

    @property
    def path(self) -> IdeaPath:
        return self.idea_path

    @property
    def file_reader(self) -> FileReaderProtocol:
        return self._file_reader

    @property
    def file_path(self) -> Path:
        return self._root_folder_path / self._relative_path

    @property
    def root_folder_path(self) -> Path:
        return self._root_folder_path

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

        If the metadata cannot be serialized, only the Markdown ``body`` is
        returned and the metadata is omitted.
        """
        return _to_text(self._metadata, self._body)

    def write(self, file_writer: FileWriterProtocol | None = None) -> None:
        """Write the serialized note to :attr:`file_path`.

        If the metadata cannot be serialized, only the Markdown body is
        written and the metadata is omitted.

        Raises:
            OSError: If ``file_writer`` cannot write the document.
        """
        file_writer = file_writer or DiskFileWriter()
        file_writer.write(self.to_text(), self.file_path)

    @property
    def link_sources(self) -> Sequence[LinkSource]:
        """Source links contained in the original text."""

        return tuple(self._link_source_extractor.extract(self.text))
