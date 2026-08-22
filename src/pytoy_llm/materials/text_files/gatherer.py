import os
from collections.abc import Sequence
from pathlib import Path

DEFAULT_EXCLUDED_PATTERNS = frozenset(
    {".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", "**/*.egg-info/**"}
)


class FilesGatherer:
    """Gathers file paths under a root directory with depth and pattern filters."""

    _ALWAYS_EXCLUDED_DIRECTORIES = frozenset(
        {
            ".git",
            "__pycache__",
        }
    )

    def __init__(self, default_excludes: frozenset[str] | tuple[str, ...] | None = None):
        if default_excludes is None:
            default_excludes = DEFAULT_EXCLUDED_PATTERNS
        self.default_excludes = frozenset(default_excludes)

    def gather(
        self,
        root: Path | str,
        max_depth: int | None = None,
        filename_patterns: tuple[str, ...] = ("*",),
        excludes: tuple[str, ...] | frozenset[str] = (),
    ) -> Sequence[Path]:
        """Gather files matching filename patterns while excluding paths from traversal.

        ``excludes`` applies to relative paths and controls directory traversal.
        ``filename_patterns`` applies only to filenames and controls collected files.
        ``max_depth`` limits the traversal depth from ``root``.

        Return:
            A sequence of absolute paths.

        """
        excludes = self.default_excludes | set(excludes)

        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"`{root=}` must be a directory.")

        paths: list[Path] = []

        for current_root, dirs, files in os.walk(root, topdown=True):
            current = Path(current_root)
            relative = current.relative_to(root)

            # directory exclusion
            dirs[:] = [directory for directory in dirs if not self._is_excluded(relative / directory, excludes, max_depth)]

            # filename inclusion
            if filename_patterns:
                files[:] = [
                    filename for filename in files if any(Path(filename).match(pattern) for pattern in filename_patterns)
                ]

            # path exclusion
            is_path_excluded = lambda path: self._matches_exclude(path.relative_to(root), excludes)
            paths += [path for filename in files if not is_path_excluded(path := (current / filename))]

        return tuple(paths)

    def _is_excluded(self, relative_path: Path, excludes: frozenset[str], max_depth: int | None) -> bool:
        return (
            self._is_always_excluded(relative_path)
            or self._matches_exclude(relative_path, excludes)
            or self._exceeds_max_depth(relative_path, max_depth)
        )

    def _matches_exclude(self, relative_path: Path, excludes: frozenset[str]) -> bool:
        return any(relative_path.match(pattern) for pattern in excludes)

    def _exceeds_max_depth(self, relative_path: Path, max_depth: int | None) -> bool:
        if max_depth is None:
            return False
        return len(relative_path.parts) > max_depth

    def _is_always_excluded(self, relative_path: Path) -> bool:
        return relative_path.name in self._ALWAYS_EXCLUDED_DIRECTORIES
