from __future__ import annotations

from pathlib import Path
from typing import Self

from pytoy_llm.materials.text_files.gatherer import FilesGatherer
from pytoy_llm.materials.text_files.models import FileMeta, TextFile, TextFilesMaterial, TextFilesMaterialQuery


class TextFilesCollector:
    def __init__(self, *, workspace: Path | str, default_excludes: None | tuple[str, ...] = None) -> None:
        workspace = Path(workspace).absolute()
        if not workspace.is_dir():
            workspace = workspace.parent
        self._workspace = Path(workspace).absolute()
        self._default_excludes = default_excludes

    @property
    def workspace(self) -> Path:
        return self._workspace

    @classmethod
    def from_inferred_workspace(cls, path: str | Path = Path(".")) -> Self:
        path = Path(path).absolute()
        start = path if path.is_dir() else path.parent
        for directory in (start, *start.parents):
            if (directory / ".git").exists():
                return cls(workspace=directory)
        return cls(workspace=start)

    @property
    def material(self) -> TextFilesMaterial:
        return self.get_material(query=TextFilesMaterialQuery(pivot=self.workspace))

    def get_material(self, query: TextFilesMaterialQuery) -> TextFilesMaterial:
        pivot_path = query.pivot
        if not pivot_path.is_dir():
            pivot_path = pivot_path.parent

        if pivot_path.is_absolute():
            root = pivot_path
        else:
            root = self.workspace / pivot_path

        if self.workspace != root and self.workspace not in root.resolve().parents:
            msg = "Specified collection root folder must be inside `workspace`."
            raise ValueError(msg)

        gatherer = FilesGatherer(default_excludes=self._default_excludes)
        file_paths = gatherer.gather(
            root, max_depth=query.max_depth, filename_patterns=query.filename_patterns, excludes=query.excludes
        )
        if query.only_meta:
            text_files = [FileMeta.from_path(path, self.workspace) for path in file_paths]
        else:
            text_files = [TextFile.from_path(path, self.workspace) for path in file_paths]
        return TextFilesMaterial(files=text_files)


if __name__ == "__main__":
    collector = TextFilesCollector(workspace=Path(__file__).parent)
    query = TextFilesMaterialQuery(max_depth=2)
    print(collector.get_material(query).text_material_data.content)
