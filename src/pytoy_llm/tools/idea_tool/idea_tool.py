from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Self, Sequence

from pytoy_llm.idea import (
    IdeaGraph,
    IdeaSpace,
    SchemeDirectory,
    UriLocalPathResolver,
)
from pytoy_llm.tools.errors import ToolError, ToolErrorKind

from .discovery import IdeaDiscovery
from .inspection import IdeaInspection
from .models import (
    IdeaSpaceToolMetaModel,
    IdeaSpaceToolWorkingContextModel,
)
from .mutation import IdeaMutation
from .semantic_types import Namespace


class IdeaTool:
    """Provides tools for accessing and modifying an IdeaSpace.

    The tool also records the start and completion times of LLM invocations.
    ``mark_llm_start`` and ``mark_llm_finished`` are intended to be registered
    as lifecycle event handlers for this purpose.

    """

    SCHEME = "idea"

    def __init__(
        self,
        idea_spaces: Mapping[Namespace, IdeaSpace] | IdeaSpace,
        local_path_resolver: UriLocalPathResolver,
        *,
        default_namespace: Namespace | None = None,
    ) -> None:

        if isinstance(idea_spaces, IdeaSpace):
            default_namespace = (
                default_namespace
                if default_namespace is not None
                else idea_spaces.root_directory_path.name
            )
            idea_spaces = {default_namespace: idea_spaces}

        if not idea_spaces:
            raise ValueError("At least one IdeaSpace must be provided.")

        if "" in idea_spaces.keys():
            raise ValueError(f"Empty `str` is not allowed for `Namespace`. `{idea_spaces=}`")

        self._idea_spaces = dict(idea_spaces)

        self._idea_graphs = {
            namespace: IdeaGraph(idea_space) for namespace, idea_space in self._idea_spaces.items()
        }

        if default_namespace is None:
            default_namespace = next(iter(self._idea_spaces))
        self._default_namespace = default_namespace

        if default_namespace not in self._idea_spaces:
            raise ValueError(f"`{default_namespace=}` does not exist in `{idea_spaces}`")

        if not local_path_resolver.is_registered(self.SCHEME, default_namespace):
            raise ValueError(f"`{default_namespace=}` is not registered.")

        if any(
            space.directory_path != space.root_directory_path
            for space in self._idea_spaces.values()
        ):
            raise ValueError(
                f"Ideaspace for this tool must be the root; However, `{idea_spaces=}`."
            )

        self._local_path_resolver = local_path_resolver
        self._discovery = IdeaDiscovery(self._get_idea_space)
        self._inspection = IdeaInspection(
            get_idea_space=self._get_idea_space,
            resolve_idea_namespace=self._resolve_idea_namespace,
            idea_graphs=self._idea_graphs,
            local_path_resolver=self._local_path_resolver,
        )
        self._mutation = IdeaMutation(self._get_idea_space)

        self._started_at: datetime = datetime.now(tz=timezone.utc)

    @classmethod
    def from_any(
        cls,
        idea_space_roots: Sequence[Path | str] | Path | str | IdeaSpace | Mapping[str, IdeaSpace],
        *,
        scheme_directories: Sequence[SchemeDirectory] = tuple(),
        workspace_root: str | Path | None = None,
        default_namespace: Namespace | None = None,
    ) -> Self:
        scheme_directories = list(scheme_directories)
        if isinstance(idea_space_roots, (Path, str)):
            idea_space_roots = [idea_space_roots]
        elif isinstance(idea_space_roots, IdeaSpace):
            idea_space_roots = [idea_space_roots.root_directory_path]

        if not isinstance(idea_space_roots, Mapping):
            roots = [Path(root).resolve() for root in idea_space_roots]
            idea_spaces = {
                root.name: IdeaSpace.from_path(path=root, root=root, with_creation=True)
                for root in roots
            }
            if len(idea_spaces) != len(roots):
                raise ValueError("IdeaSpace roots must have unique directory names.")
        else:
            idea_spaces = idea_space_roots

        if workspace_root is not None:
            workspace_root = Path(workspace_root).resolve()
            if any(elem.scheme == "workspace" for elem in scheme_directories):
                raise ValueError(
                    f"Dumplication of `workspace` scheme, {scheme_directories=}, {workspace_root=} "
                )
            workspace_scheme_directory = SchemeDirectory(
                root_directory=workspace_root, scheme="workspace"
            )
            scheme_directories.append(workspace_scheme_directory)

        for name, space in idea_spaces.items():
            scheme_directories.append(
                SchemeDirectory(
                    root_directory=space.root_directory_path, scheme=cls.SCHEME, authority=name
                )
            )

        local_path_resolver = UriLocalPathResolver.from_any(
            scheme_directories=scheme_directories,
        )

        return cls(
            idea_spaces=idea_spaces,
            local_path_resolver=local_path_resolver,
            default_namespace=default_namespace,
        )

    @property
    def default_namespace(self) -> Namespace:
        return self._default_namespace

    @property
    def inspection(self) -> IdeaInspection:
        return self._inspection

    @property
    def discovery(self) -> IdeaDiscovery:
        return self._discovery

    @property
    def mutation(self) -> IdeaMutation:
        return self._mutation

    def _resolve_idea_namespace(
        self, idea_namespace: Namespace | None = None
    ) -> Namespace | ToolError:
        namespace = self.default_namespace if idea_namespace is None else idea_namespace
        if namespace not in self._idea_spaces:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg=f"The namespace `{namespace}` does not exist.",
                suggestion="Call `get_idea_namespaces()` to list available namespaces.",
            )
        return namespace

    def _get_idea_space(self, idea_namespace: Namespace | None = None) -> IdeaSpace | ToolError:
        resolved_idea_namespace = self._resolve_idea_namespace(idea_namespace)
        if isinstance(resolved_idea_namespace, ToolError):
            return resolved_idea_namespace
        return self._idea_spaces[resolved_idea_namespace]

    def get_ideaspace_root(self, idea_namespace: Namespace | None = None) -> Path | ToolError:
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        return idea_space.root_directory_path

    def get_ideaspace(self, idea_namespace: Namespace | None = None) -> IdeaSpace | ToolError:
        return self._get_idea_space(idea_namespace)

    def get_tool_context_path(self, idea_namespace: Namespace | None = None) -> Path | ToolError:
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        return idea_space.space_meta_directory / "tool_context.json"

    @property
    def tools(self) -> Sequence[Callable]:
        tools = [
            self.get_idea_namespaces,
            self.get_default_idea_namespace,
            self.get_idea_space_working_context,
            *self._discovery.tools,
            *self._inspection.tools,
            *self._mutation.tools,
        ]
        return tools

    def mark_llm_start(self) -> None:
        """Mark the start of an LLM invocation.

        Intended to be subscribed to the invocation start event.
        """
        self._started_at = datetime.now(tz=timezone.utc)

    def mark_llm_finished(self) -> None:
        """Mark the completion of an LLM invocation.

        Intended to be subscribed to the invocation completion event.
        """
        model = IdeaSpaceToolMetaModel.model_validate(
            {
                "last_llm_started_at": self._started_at,
                "last_llm_finished_at": datetime.now(tz=timezone.utc),
            }
        )
        for namespace in self._idea_spaces:
            context_path = self.get_tool_context_path(namespace)
            if isinstance(context_path, Path):
                context_path.parent.mkdir(exist_ok=True, parents=True)
                context_path.write_text(model.model_dump_json(indent=2), encoding="utf8")

    def get_idea_namespaces(self) -> Sequence[Namespace]:
        """Get the available namespaces for the IdeaSpace."""

        return list(self._idea_spaces.keys())

    def get_default_idea_namespace(self) -> Namespace:
        """Get the default namespace for the IdeaSpace."""
        return self._default_namespace

    def get_idea_space_working_context(
        self, idea_namespace: Namespace | None = None
    ) -> IdeaSpaceToolWorkingContextModel | ToolError:
        """Get the current tool-related working context for the IdeaSpace.

        The working context records the start and completion times of the latest
        LLM interaction. A missing context file is treated as an initialized
        context with both timestamps set to ``null``; it is not an error.

        Args:
            idea_namespace:
                Namespace of the IdeaSpace. ``null`` uses the default namespace.

        Returns:
            ``IdeaSpaceToolWorkingContextModel`` containing the latest recorded
            interaction timestamps. Both timestamps are ``null`` when no interaction
            has been recorded.

            ``ToolError`` if the saved context exists but cannot be parsed.
        """
        context_path = self.get_tool_context_path(idea_namespace)
        if isinstance(context_path, ToolError):
            return context_path

        try:
            text = context_path.read_text()
        except FileNotFoundError:
            tool_meta = IdeaSpaceToolMetaModel()
        else:
            try:
                tool_meta = IdeaSpaceToolMetaModel.model_validate_json(text)
            except ValueError as exc:
                return ToolError(
                    kind=ToolErrorKind.UNKNOWN,
                    msg=f"`IdeaSpaceToolMetaModel` cannot be made: {exc}",
                )
        return IdeaSpaceToolWorkingContextModel(idea_space_tool_meta=tool_meta)
