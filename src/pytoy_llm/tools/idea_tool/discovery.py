from typing import Callable, Sequence

from pytoy_llm.idea import IdeaSpace, ShouldBeSpacePathError
from pytoy_llm.tools.errors import ToolError, ToolErrorKind

from .boundaries import tool_discovery_boundary
from .semantic_types import (
    IdeaNotePath,
    IdeaSpaceDepth,
    IdeaSpacePath,
    IdeaSpacePivot,
    Namespace,
)


class IdeaDiscovery:
    """Provide read-only discovery tools for IdeaSpace paths and conventions."""

    def __init__(
        self,
        get_idea_space: Callable[[Namespace | None], IdeaSpace | ToolError],
    ) -> None:
        self._get_idea_space = get_idea_space

    @property
    def tools(self) -> Sequence[Callable]:
        return [
            self.get_all_sub_idea_spaces_supported_by_convention,
            self.get_sub_idea_spaces,
            self.get_idea_note_paths,
        ]

    @tool_discovery_boundary
    def get_all_sub_idea_spaces_supported_by_convention(
        self,
        idea_namespace: Namespace | None = None,
    ) -> Sequence[IdeaSpacePath] | ToolError:
        """List all IdeaSpace paths where a convention is defined.

        The returned paths are relative to the selected IdeaSpace root.
        The root is represented by ``.``.

        Use each returned path as the ``idea_space_path`` argument of
        ``get_idea_space_convention`` to read the convention that applies to that path.
        This tool returns IdeaSpace paths only; it does not return convention contents.

        Args:
            idea_namespace:
                Namespace of the IdeaSpace. ``null`` uses the default namespace.

        Returns:
            A sequence of IdeaSpace-relative pivots sorted by path. An empty sequence
            means that no convention is defined in the IdeaSpace.

            ``ToolError`` if the IdeaSpace cannot be inspected.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        spaces = [idea_space, *idea_space.get_subspaces(depth=None)]
        return sorted(space.idea_path for space in spaces if space.convention is not None)

    @tool_discovery_boundary
    def get_sub_idea_spaces(
        self,
        idea_space_pivot: IdeaSpacePivot = ".",
        depth: IdeaSpaceDepth = 0,
        idea_namespace: Namespace | None = None,
    ) -> Sequence[IdeaSpacePath] | ToolError:
        """List IdeaSpace subdirectories below ``idea_space_pivot``.

        Args:
            depth:
                ``0`` returns only immediate child subspaces. ``null``
                returns subspaces at all descendant levels.

        Returns:
            IdeaSpace-root-relative paths of matching subspaces. This tool
            returns paths only.

            ``ToolError`` if ``idea_space_pivot`` is outside the IdeaSpace or cannot be
            inspected.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        path = idea_space.resolve(idea_space_pivot)
        try:
            sub_space = IdeaSpace.from_path(path, root=idea_space.root_directory_path)
        except ShouldBeSpacePathError as exc:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=str(exc))
        result_spaces = sub_space.get_subspaces(depth=depth)
        return [space.idea_path for space in result_spaces]

    @tool_discovery_boundary
    def get_idea_note_paths(
        self,
        idea_space_pivot: IdeaSpacePivot = ".",
        depth: IdeaSpaceDepth = 0,
        idea_namespace: Namespace | None = None,
    ) -> Sequence[IdeaNotePath] | ToolError:
        """List IdeaNote paths under the specified IdeaSpace pivot.

        Args:
            idea_space_pivot:
                Starting IdeaSpace path. ``.`` represents the IdeaSpace root.
            depth:
                ``0`` returns notes directly under the pivot.
                ``null`` returns notes at all descendant levels.
            idea_namespace:
                Namespace of the IdeaSpace. ``null`` uses the default namespace.

        Returns:
            IdeaSpace-root-relative paths of matching IdeaNotes.
            The result is empty when no matching IdeaNotes exist.

            ``ToolError`` if the IdeaSpace cannot be inspected.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        pivot_path = idea_space.resolve(idea_space_pivot)
        try:
            result_notes = IdeaSpace.from_path(
                pivot_path, root=idea_space.root_directory_path
            ).get_notes(depth=depth)
        except ShouldBeSpacePathError as exc:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=str(exc))
        return [note.idea_path for note in result_notes]
