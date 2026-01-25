from pathlib import Path
from pytoy_llm.materials.git_diffs.models import LineRange
from typing import Sequence, Callable, Any
from git import Repo, Diff, DiffIndex
from pytoy_llm.materials.git_diffs.models import (
    FileAdd,
    FileDelete,
    FileModify,
    FileDiff,
    AtomicChange,
    FileOperation,
    DiffBundle,
    GitDiffBundleQuery
)
import re


# --- HunkState (1 hunk = 1 AtomicChange) ---
class _HunkState:
    def __init__(self, start_line: int):
        self.start_line = start_line
        self.old_lines: list[str] = []
        self.new_lines: list[str] = []

    def add_delete(self, line: str):
        self.old_lines.append(line)

    def add_add(self, line: str):
        self.new_lines.append(line)

    def to_atomic_change(self) -> AtomicChange:
        return AtomicChange(
            range=LineRange(start=self.start_line, end=self.start_line + len(self.old_lines)),
            old_lines=self.old_lines,
            new_lines=self.new_lines,
        )


def _to_str(arg: str | bytes) -> str:
    if isinstance(arg, str):
        return arg
    return arg.decode("utf-8", errors="ignore")  # type: ignore


class FileAddCreator:
    def __call__(self, diff: Diff) -> FileAdd:
        inner = diff.diff
        assert inner is not None

        lines: list[str] = []
        for line in _to_str(inner).splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                lines.append(line[1:])
        path = diff.b_path
        assert path
        return FileAdd(path=Path(path), lines=lines)


class FileDeleteCreator:
    def __call__(self, diff: Diff) -> FileDelete:
        inner = diff.diff
        assert inner is not None

        old_lines: list[str] = []
        for line in _to_str(inner).splitlines():
            if line.startswith("-") and not line.startswith("---"):
                old_lines.append(line[1:])
        path = diff.a_path
        assert path
        return FileDelete(path=Path(path), old_lines=old_lines)


class FileModifyCreator:
    HUNK_RE = re.compile(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@")

    def __call__(self, diff: Diff) -> FileModify:
        diff_text = _to_str(diff.diff or b"").splitlines()
        atomic_changes: list[AtomicChange] = []

        def process_hunk(hunk_lines: list[str], start_line: int) -> AtomicChange:
            """1つのhunkを解析してAtomicChangeを返す"""
            hunk_state = _HunkState(start_line=start_line)

            for line in hunk_lines[1:]:  # 最初の @@ はスキップ
                if line.startswith("+") and not line.startswith("+++"):
                    hunk_state.add_add(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    hunk_state.add_delete(line[1:])
                else:
                    # context line は old_line だけ進めるが flush は不要
                    pass
            return hunk_state.to_atomic_change()

        i = 0
        while i < len(diff_text):
            line = diff_text[i]
            if line.startswith("@@"):
                match = self.HUNK_RE.match(line)
                if match:
                    old_line = int(match.group(1)) - 1  # 0-based index
                    # hunk の終わりまで切り出し
                    j = i + 1
                    while j < len(diff_text) and not diff_text[j].startswith("@@"):
                        j += 1
                    hunk_lines = diff_text[i:j]
                    ac = process_hunk(hunk_lines, old_line)
                    if ac.old_lines or ac.new_lines:
                        atomic_changes.append(ac)
                    i = j
                    continue
                else:
                    raise RuntimeError(f"Cannot identify `{diff_text=}`")
            i += 1
        path = diff.b_path or diff.a_path
        assert path, "path must be existed."
        return FileModify(path=Path(path), atomic_changes=atomic_changes)


class FileOperationCreator:
    def __init__(self) -> None:
        # Other `diff.change_type` exist.
        self._creators: dict[str | None, Callable[[Diff], FileOperation]] = {
            "A": FileAddCreator(),
            "D": FileDeleteCreator(),
            "M": FileModifyCreator(),
            None: FileModifyCreator(),
        }

    def __call__(self, diff_index: DiffIndex) -> Sequence[FileOperation]:
        file_operations = []
        for diff in diff_index:
            if creator := self._creators.get(diff.change_type):
                file_operations.append(creator(diff))
            else:
                pass
        return file_operations
    

class GitDiffCollector:
    def __init__(
        self, repo_path: str | Path | None = None, root_folder: str | Path | None = None
    ) -> None:
        # Note: `repo_path should be a directory.
        if repo_path is None:
            repo_path = Path(".")
        repo_path = Path(repo_path)
        self.repo_path = repo_path if repo_path.is_dir() else repo_path.parent
        self.repo = Repo(self.repo_path, search_parent_directories=True)
        self.workspace = Path(self.repo.git_dir).parent
        self.root_folder = Path(root_folder) if root_folder else self.workspace

        if not self.root_folder.is_relative_to(self.workspace):
            raise ValueError(f"`{self.root_folder}` must be inside `workspace`.")
        self.root_location: Sequence[str] = self.root_folder.relative_to(self.workspace).parts

        self.operation_creator = FileOperationCreator()
        
    def get_bundle(self, query: GitDiffBundleQuery) -> DiffBundle:
        from_rev = query.from_rev
        to_rev = query.to_rev
        
        if from_rev == "index":
            base = self.repo.index
        else:
            from_rev = from_rev or "HEAD"
            base = self.repo.commit(from_rev)
            
        if to_rev == "index":
            diffs = base.diff(create_patch=True)
            timestamp_provider = lambda path: path.stat().st_mtime if path.exists() else -1 # noqa: E731
        elif to_rev == "working-tree":
            diffs = base.diff(None, create_patch=True)
            timestamp_provider = lambda path: path.stat().st_mtime if path.exists() else -1 # noqa: E731
        else:
            to_rev = to_rev or "HEAD"
            diffs = base.diff(to_rev, create_patch=True)
            timestamp = self.repo.commit(to_rev).committed_date
            timestamp_provider = lambda _: float(timestamp) # noqa: E731
        return self._create_bundle_from_ops(diffs, timestamp_provider)
    

    
    def _create_bundle_from_ops(self, diffs: Any, timestamp_provider: Callable[[Path], float]) -> DiffBundle:
        ops = self.operation_creator(diffs)
        file_diffs = [
            FileDiff(
                operation=op,
                timestamp=timestamp_provider(op.path),
                location=op.path.relative_to(self.workspace).parts,
            )
            for op in ops
        ]
        return DiffBundle(root_location=self.root_location, file_diffs=file_diffs)

    @property
    def bundle(self) -> DiffBundle:
        return self.get_bundle(GitDiffBundleQuery(from_rev="head", to_rev="working-tree"))



if __name__ == "__main__":
    pass
