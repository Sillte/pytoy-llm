import itertools
import re
from pathlib import Path
from typing import Annotated, Self, Sequence

from pydantic import Field

from pytoy_llm.foundation.paths import PathGatherer
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer.models import GrepMatch, GrepMatchContext, WorkspaceAccess
from pytoy_llm.tools.workspace_explorer.semantic_types import GlobPattern, SearchPattern, WorkspacePath


class WorkspaceSearch:
    """
    Provide safe workspace search tools for LLM agents.

    This class provides read-only operations for searching the contents
    of files within the workspace.
    It does not modify workspace files.

    Every path is interpreted relative to the workspace root.
    Files outside the workspace are never accessible.
    """

    def __init__(self, access: WorkspaceAccess) -> None:
        self.access = access
        self.workspace = access.workspace.resolve()

    @classmethod
    def from_any(cls, workspace: Path | str, excludes: frozenset[str] | None = None) -> Self:
        return cls(access=WorkspaceAccess.from_any(workspace=workspace, excludes=excludes))

    @property
    def tools(
        self,
    ):
        return [self.grep_context]

    def grep_context(
        self,
        search_patterns: Annotated[
            SearchPattern | Sequence[SearchPattern],
            Field(description="Search patterns are combined with OR semantics."),
        ],
        collection_root: WorkspacePath = ".",
        file_patterns: Annotated[
            GlobPattern | Sequence[GlobPattern],
            Field(description="Glob patterns matched against paths relative to `collection_root`."),
        ] = ("*",),
        regex: Annotated[
            bool,
            Field(description="Interpret the search pattern as a regular expression."),
        ] = False,
        case_sensitive: Annotated[
            bool,
            Field(description="Whether the search is case-sensitive."),
        ] = False,
        context_lines: Annotated[
            int,
            Field(
                ge=0,
                description=("Number of lines to include before and after each match."),
            ),
        ] = 3,
    ) -> Sequence[GrepMatchContext] | ToolError:
        """
        Search text contents of files within the workspace and return matching
        lines together with surrounding context.

        Each search pattern is matched independently, and multiple patterns are
        combined with OR semantics. By default, patterns are treated as literal
        text. When `regex` is `True`, patterns are interpreted as regular
        expressions.

        Searches are restricted to files under `collection_root` and can be
        further limited by `file_patterns`. Matching results are grouped by file
        and nearby matches are combined into context ranges according to
        `context_lines`.

        Args:
            search_patterns:
                One or more text strings or regular expressions to search for.
                Multiple patterns are combined with OR semantics.

            collection_root:
                Directory relative to the workspace root from which the search
                starts.

            file_patterns:
                Glob pattern or glob patterns used to select files relative to `collection_root`.
                If omitted, all files are searched.

            regex:
                If `False`, search patterns are matched as literal text.
                If `True`, search patterns are interpreted as regular expressions.

            case_sensitive:
                Whether text or regular expression matching is case-sensitive.

            context_lines:
                Number of lines to include before and after each matching line.
                Overlapping or adjacent context ranges are merged.

        Returns:
            A sequence of `GrepMatchContext` objects containing the matching file,
            the surrounding content, the line range, and the individual matches.

            Returns `ToolError` when the collection root is invalid or the search
            cannot be performed.
        """
        if isinstance(search_patterns, str):
            search_patterns = [search_patterns]
        if isinstance(file_patterns, str):
            file_patterns = [file_patterns]

        root = self.access.resolve(collection_root)
        if isinstance(root, ToolError):
            return root
        if not root.exists():
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=f"`{root=}` does not exist.", retry=None)
        if not root.is_dir():
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=f"`{root=}` must be a directory.", retry=None)

        files = PathGatherer().gather(
            root=root,
            patterns=file_patterns,
            excludes=self.access.excludes,
            target="file",
        )
        if regex:
            matches = self._grep_context_by_regex(
                search_patterns=search_patterns,
                files=files,
                case_sensitive=case_sensitive,
            )
        else:
            matches = self._grep_context_by_text(search_patterns=search_patterns, files=files, case_sensitive=case_sensitive)
        if isinstance(matches, ToolError):
            return matches
        return self.integrate_matches(matches, context_lines=context_lines)

    def _grep_context_by_regex(
        self,
        search_patterns: Annotated[
            Sequence[SearchPattern], Field(description="Search patterns are combined with OR semantics.")
        ],
        files: Sequence[Path],
        case_sensitive: Annotated[
            bool,
            Field(description="Whether the search is case-sensitive."),
        ] = False,
    ):
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            compilations = [re.compile(pattern, flags) for pattern in search_patterns]
        except re.error as exc:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg=str(exc),
                retry=False,
            )
        return tuple(
            itertools.chain.from_iterable((self._get_matched_by_regex(file_path, compilations) for file_path in files))
        )

    def _get_matched_by_regex(self, file_path: Path, patterns: Sequence[re.Pattern[str]]) -> Sequence[GrepMatch]:
        """
        file_path: The abolute path
        """
        try:
            lines = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except OSError:
            return []

        results = []

        for lineno, line in enumerate(lines):
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    results.append(
                        GrepMatch(
                            path=file_path.relative_to(self.workspace).as_posix(),
                            line=lineno,
                            column=match.start(),
                            text=line,
                        )
                    )
                    break
        return results

    def _grep_context_by_text(
        self,
        search_patterns: Sequence[SearchPattern],
        files: Sequence[Path],
        case_sensitive: bool = False,
    ) -> Sequence[GrepMatch]:

        return tuple(
            itertools.chain.from_iterable(
                self._get_matched_by_text(file_path, patterns=search_patterns, case_sensitive=case_sensitive)
                for file_path in files
            )
        )

    def _get_matched_by_text(
        self,
        file_path: Path,
        patterns: Sequence[SearchPattern],
        case_sensitive: bool = False,
    ) -> Sequence[GrepMatch]:

        try:
            content = file_path.read_text(encoding="utf8")
            lines = content.splitlines(keepends=True)
        except OSError:
            return []

        results = []

        for lineno, line in enumerate(lines):
            source = line if case_sensitive else line.lower()
            for pattern in patterns:
                target = pattern if case_sensitive else pattern.lower()

                idx = source.find(target)
                if idx != -1:
                    results.append(
                        GrepMatch(
                            path=file_path.relative_to(self.workspace).as_posix(),
                            line=lineno,
                            column=idx,
                            text=line,
                        )
                    )
                    break
        return results

    def integrate_matches(self, matches: Sequence[GrepMatch], context_lines: int) -> Sequence[GrepMatchContext] | ToolError:
        matches_by_path = {}
        for match in matches:
            matches_by_path.setdefault(match.path, []).append(match)

        contexts: list[GrepMatchContext] = []

        for match_path, file_matches in matches_by_path.items():
            file_path = self.workspace / match_path
            result = _build_context(file_path, file_matches, context_lines)
            if isinstance(result, ToolError):
                return result
            contexts.extend(result)
        return contexts


def _build_context(file_path: Path, matches: Sequence[GrepMatch], context_lines: int) -> Sequence[GrepMatchContext] | ToolError:
    if not matches:
        return []

    matches = sorted(matches, key=lambda match: match.line)
    workspace_path = matches[0].path

    try:
        content = file_path.read_text(encoding="utf8")
        lines = content.splitlines(keepends=True)
    except OSError:
        return ToolError(kind=ToolErrorKind.RESOURCE_LIMIT, msg="Resource usage is high. Please restrict the scope of search.")
    # `start_line`, `end_line`, list[GrepMatch]
    ranges: list[tuple[int, int, list[GrepMatch]]] = []
    contexts: list[GrepMatchContext] = []

    for match in matches:
        start_line = max(0, match.line - context_lines)
        end_line = min(len(lines), match.line + context_lines + 1)

        ranges.append(
            (
                start_line,
                end_line,
                [match],
            )
        )

    # Merge overlapping or adjacent ranges.
    merged: list[tuple[int, int, list[GrepMatch]]] = [ranges[0]]
    for start_line, end_line, range_matches in ranges[1:]:
        previous_start, previous_end, previous_matches = merged[-1]

        if start_line <= previous_end:
            merged[-1] = (
                previous_start,
                max(previous_end, end_line),
                previous_matches + range_matches,
            )
        else:
            merged.append(
                (
                    start_line,
                    end_line,
                    range_matches,
                )
            )

    # Read each merged range.
    for start_line, end_line, range_matches in merged:
        content = "".join(lines[start_line:end_line])
        contexts.append(
            GrepMatchContext(
                path=workspace_path,
                content=content,
                start_line=start_line,
                end_line=end_line,
                matches=range_matches,
            )
        )
    return contexts
