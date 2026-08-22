from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

DEFAULT_EXCLUDED_PATTERNS = frozenset(
    {".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", "**/*.egg-info/**"}
)


class PathGatherer:
    """Gathers paths under a root directory with depth and pattern filters."""

    _ALWAYS_EXCLUDED_DIRECTORIES = frozenset(
        {
            ".git",
            "__pycache__",
        }
    )

    def __init__(self, default_excludes: frozenset[str] | Sequence[str] | None = None):
        if default_excludes is None:
            default_excludes = DEFAULT_EXCLUDED_PATTERNS
        self.default_excludes = frozenset(default_excludes)

    def gather(
        self,
        root: Path | str,
        max_depth: int | None = None,
        patterns: Sequence[str] = ("*",),
        excludes: Sequence[str] | frozenset[str] = (),
        target: Literal["all", "directory", "file"] = "all",
    ) -> Sequence[Path]:
        """Gather files matching filename patterns while excluding paths from traversal.

        ``patterns`` applies to the candidates of the elements of the return. If empty, all non-excluded paths are returned.
        ``excludes`` applies to relative paths and controls directory traversal.
        ``max_depth`` limits the traversal depth from ``root``.
        ``target``: the kind of `paths` which are returned.

        Return:
            A sequence of absolute paths.

        """
        if not patterns:
            patterns = ("*",)

        def _is_pattern_matched(relative_path: Path) -> bool:
            return any(relative_path.match(pattern) for pattern in patterns)

        excludes = self.default_excludes | set(excludes)

        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"`{root=}` must be a directory.")

        paths: list[Path] = []

        for current_root, dirs, files in os.walk(root, topdown=True):
            current = Path(current_root)

            # directory exclusion
            dirs[:] = [
                directory
                for directory in dirs
                if not self._is_excluded((current / directory).relative_to(root), excludes, max_depth)
            ]

            # path exclusion
            def is_path_target(relative_path: Path) -> bool:
                return not self._matches_exclude(relative_path, excludes) and _is_pattern_matched(relative_path)

            # file path
            if target == "file" or target == "all":
                file_paths = [current / filename for filename in files]
                paths += [path for path in file_paths if is_path_target(path.relative_to(root))]

            # directory path
            if target in {"all", "directory"}:
                directory_paths = [current / directory for directory in dirs]
                paths += [path for path in directory_paths if is_path_target(path.relative_to(root))]

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
