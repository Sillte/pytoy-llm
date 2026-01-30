from pathlib import Path
from pydantic import BaseModel
from typing import Sequence, Self
from datetime import datetime
import time
from pytoy_llm.materials.utils import FileGatherer
from pytoy_llm.materials.text_files.models import TextFileBundle, TextFileBundleQuery, TextFileLocator, TextFileCollection, TextFileContent


class TextFilesCollector:
    def __init__(self,
                 start_path: str | Path,
                 workspace: str | Path | None = None)  -> None:
        start_path = Path(start_path)
        workspace = workspace or start_path.parent 
        self._start_path = start_path
        self._workspace = Path(workspace)
        self._cwd = Path(start_path).parent
        self._ext = start_path.suffix
        if start_path.is_dir():
            raise ValueError(f"`{start_path=}` must be a filepath, not directory.")
        if not start_path.is_relative_to(workspace):
            raise ValueError(f"`{start_path=}` is not a decendant of `{workspace=}`")

    @property
    def bundle(self) -> TextFileBundle:
        query = TextFileBundleQuery()
        return self.get_bundle(query)
    
    def get_bundle(self, query: TextFileBundleQuery) -> TextFileBundle:
        _ = query # Todo: parameters of`FileGatherer` can be set by `TextFileBundleQueyr` in the future.
        paths = FileGatherer().gather(self._cwd)
        paths = [path for path in paths if path.suffix == self._ext]
        locator_list = [TextFileLocator.from_path(path, self._workspace) for path in paths]
        locators = {item.id: item for item in locator_list}
        contents = {id_: TextFileContent.from_locator(locator, self._workspace) for id_, locator in locators.items()}
        collection = TextFileCollection(locators=locators, contents=contents)
        return TextFileBundle(collection=collection, )  # Other property....




if __name__ == "__main__":
    pass
