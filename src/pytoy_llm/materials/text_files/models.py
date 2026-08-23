from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, BeforeValidator, Field

from pytoy_llm.foundation.paths import PathTree
from pytoy_llm.materials.models import ModelMaterialData, TextMaterialData


def check_relative_path(v: Any) -> Path:
    p = Path(v)
    if p.is_absolute():
        raise ValueError("Path must be relative")
    return p


TextFilePath = Annotated[Path, Field(description="Relative path"), BeforeValidator(check_relative_path)]


class TextFile(BaseModel, frozen=True):
    """Represents a collected text file within a workspace."""

    path: Annotated[
        TextFilePath,
        Field(description="Path relative to the workspace root"),
    ]

    modified_at: float = Field(..., description="Last modification time of the file (epoch seconds)")
    body: str = Field(..., description="The content of the file")

    @classmethod
    def from_path(cls, abs_path: str | Path, workspace: Path) -> Self:
        abs_path = Path(abs_path)
        body = abs_path.read_text(encoding="utf8")
        relative = Path(abs_path).relative_to(workspace)
        return cls(modified_at=abs_path.stat().st_mtime, path=relative, body=body)

    @property
    def structured_text(self) -> str:
        lines = []
        lines.append(f"<entry path={self.path.as_posix()} modified_at={self.modified_at}>")
        lines.append("<<<BEGIN>>>")
        lines.append(self.body)
        lines.append("<<<END>>>")
        lines.append("</entry>\n")
        return "\n".join(lines)


class FileMeta(BaseModel, frozen=True):
    """Represents a collected file meta within a workspace."""

    path: Annotated[
        TextFilePath,
        Field(description="Path relative to the workspace root"),
    ]

    modified_at: float = Field(..., description="Last modification time of the file (epoch seconds)")
    size: int = Field(..., description="Byte size of the text")

    @classmethod
    def from_path(cls, abs_path: str | Path, workspace: Path) -> Self:
        abs_path = Path(abs_path)
        relative = Path(abs_path).relative_to(workspace)
        stat = abs_path.stat()
        return cls(modified_at=stat.st_mtime, path=relative, size=stat.st_size)

    @property
    def structured_text(self) -> str:
        lines = []
        lines.append(f"<entry path={self.path.as_posix()} modified_at={self.modified_at} size={self.size} />")
        return "\n".join(lines)


class TextFilesMaterial(BaseModel, frozen=True):
    files: Annotated[
        Sequence[TextFile | FileMeta],
        Field(description="Files under the workspace"),
    ]

    @property
    def tree(self) -> str:
        return self._make_tree()

    @property
    def text_material_data(self) -> TextMaterialData:
        structured_text = self._make_structured_text(with_tree=True)
        description = "Collection of text files"
        return TextMaterialData(content=structured_text, description=description)

    @property
    def model_material_data(self) -> ModelMaterialData:
        return ModelMaterialData(description="Collection of text files", instances=self.files)

    def _make_structured_text(self, with_tree: bool = False) -> str:
        lines = []
        if with_tree:
            lines.extend(
                [
                    "===Tree (Path Overview)===",
                    self._make_tree(),
                    "",
                ]
            )
        lines += ["===Tag Description==="]
        lines.append("* Tag info:")
        lines.append("  - path: Relative path to the workspace")
        lines.append("  - modified_at: Last modification time of the file (epoch seconds)")
        lines.append("  - size: File size in bytes (FileMeta only)")
        lines.append("")
        lines.append("* entry:")
        lines.append("  - TextFile entries include the file body.")
        lines.append("  - FileMeta entries include metadata only; the body is not loaded.")
        lines.append("")
        lines.append("* body: The actual content of the file")
        lines.append("  - Present only for TextFile entries.")
        lines.append("  - body is wrapped between <<<BEGIN>>> and <<<END>>>")
        lines.append("")
        lines.append("\n")

        lines += ["===Instances==="]
        for f in self.files:
            lines.append(f.structured_text)
        return "\n".join(lines)

    def _make_tree(self) -> str:
        paths = [file.path for file in self.files]
        tree = PathTree.from_paths(paths=paths, root_path=Path("."))
        return tree.render(include_root=False)


class TextFilesMaterialQuery(BaseModel, frozen=True):
    """

    Note: -Specification of glob patterns-
    - `patterns` uses `pathlib.PurePath.match` semantics.
    - `patterns` filters collected target paths at addition.
    - `excludes` prevents excluded paths from being traversed or returned.
    """

    collection_root: Path = Field(
        default=Path("."),
        description="Collection root path of the material. It is either absolute path or relative path to the workspace",
    )
    max_depth: int | None = Field(
        default=None,
        ge=0,
        description="Maximum directory depth relative to the collection root.",
    )

    only_meta: bool = Field(default=False, description="If `True`, file bodies are not loaded.")

    patterns: Sequence[str] = Field(
        default=(),
        description="Glob patterns of filenames to include. If empty, all files are included.",
    )

    excludes: Sequence[str] = Field(
        default=(),
        description="Glob patterns of paths to exclude.",
    )

    @classmethod
    def from_any(
        cls,
        collection_root: Path,
        max_depth: int | None,
        only_meta: bool = True,
        patterns: str | Sequence[str] = (),
        excludes: str | Sequence[str] = (),
    ) -> Self:
        if isinstance(patterns, str):
            patterns = [
                patterns,
            ]
        if isinstance(excludes, str):
            excludes = [
                excludes,
            ]
        return cls(
            collection_root=collection_root, max_depth=max_depth, only_meta=only_meta, patterns=patterns, excludes=excludes
        )
