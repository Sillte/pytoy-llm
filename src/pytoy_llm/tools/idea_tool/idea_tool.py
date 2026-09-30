from datetime import datetime, timezone
from pathlib import Path
from posixpath import normpath
from typing import Callable, Mapping, Self, Sequence
from urllib.parse import unquote

from pydantic import AwareDatetime

from pytoy_llm.idea import (
    DiskFileWriter,
    IdeaGraph,
    IdeaNote,
    IdeaSpace,
    MetadataDeserializationError,
    MetadataValueError,
    OutsidePathError,
)
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer import WorkspaceExplorer

from .boundaries import tool_discovery_boundary, tool_inspection_boundary
from .discovery import IdeaDiscovery
from .models import (
    IdeaNoteLinkModel,
    IdeaNoteModel,
    IdeaSpaceConventionModel,
    IdeaSpaceToolMetaModel,
    IdeaSpaceToolWorkingContextModel,
    RemoteLinkModel,
    UnresolvedLinkModel,
    WorkspaceLinkModel,
)
from .semantic_types import (
    IdeaNoteBody,
    IdeaNoteMetadata,
    IdeaNotePath,
    IdeaNoteReference,
    IdeaSpaceDepth,
    IdeaSpacePath,
    IdeaSpacePivot,
    Namespace,
)


def build_idea_note_model(
    idea_note: IdeaNote,
    idea_graph: IdeaGraph,
    workspace_root: Path | None,
    namespace: Namespace,
) -> IdeaNoteModel | ToolError:
    idea_links = idea_graph.resolve_links(idea_note)

    idea_note_links = []
    workspace_links = []
    remote_links = []
    unresolved_links = []

    source_idea_note_reference = IdeaNoteReference(
        idea_note_path=idea_note.idea_path, namespace=namespace
    )

    for idea_link in idea_links:
        try:
            if idea_link.uri.scheme == "idea":
                target_namespace = idea_link.uri.authority or namespace
                idea_note_links.append(
                    IdeaNoteLinkModel(
                        source_idea_note_reference=source_idea_note_reference,
                        target_idea_note_reference=IdeaNoteReference(
                            idea_note_path=idea_link.uri.path,
                            namespace=target_namespace,
                        ),
                    )
                )
            elif idea_link.uri.scheme == "workspace":
                if workspace_root is not None:
                    workspace_path = _relative_path_from_uri(
                        idea_link.uri.path, idea_link.uri.authority
                    )
                    workspace_links.append(
                        WorkspaceLinkModel(
                            source_idea_note_reference=source_idea_note_reference,
                            workspace_file_path=workspace_path,
                        ),
                    )
                else:
                    raise ValueError("Workspace is not given here.")
            else:
                remote_links.append(
                    RemoteLinkModel(
                        source_idea_note_reference=source_idea_note_reference,
                        uri=idea_link.uri,
                    )
                )
        except (ValueError, TypeError) as exc:
            unresolved_links.append(
                UnresolvedLinkModel(
                    source_idea_note_reference=source_idea_note_reference,
                    uri=idea_link.uri,
                    reason=str(exc),
                )
            )

    modified_at = datetime.fromtimestamp(
        idea_note.file_path.stat().st_mtime,
        timezone.utc,
    )
    return IdeaNoteModel(
        idea_note_path=idea_note.idea_path,
        modified_at=modified_at,
        body=idea_note.body,
        metadata=idea_note.metadata.as_dict(),
        idea_note_links=idea_note_links,
        remote_links=remote_links,
        workspace_local_links=workspace_links,
        unresolved_links=unresolved_links,
    )


def _relative_path_from_uri(path: str, authority: str) -> str:
    if authority:
        raise ValueError(f"URI authority is not supported: {authority}")

    normalized_path = normpath(unquote(path).lstrip("/"))
    if normalized_path == ".." or normalized_path.startswith("../"):
        raise ValueError("URI path must stay relative to its registered root.")
    return normalized_path


class IdeaTool:
    """Provides tools for accessing and modifying an IdeaSpace.

    The tool also records the start and completion times of LLM invocations.
    ``mark_llm_start`` and ``mark_llm_finished`` are intended to be registered
    as lifecycle event handlers for this purpose.

    Note that this class may also provide the tools of WorkspaceExplorer.

    """

    def __init__(
        self,
        idea_spaces: Mapping[Namespace, IdeaSpace] | IdeaSpace,
        workspace_explorer: WorkspaceExplorer | None = None,
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
        if any(
            space.directory_path != space.root_directory_path
            for space in self._idea_spaces.values()
        ):
            raise ValueError(
                f"Ideaspace for this tool must be the root; However, `{idea_spaces=}`."
            )

        self._file_writer = DiskFileWriter()
        self._workspace_explorer = workspace_explorer
        self._discovery = IdeaDiscovery(self._get_idea_space)

        self._started_at: datetime = datetime.now(tz=timezone.utc)

    @property
    def default_namespace(self) -> Namespace:
        return self._default_namespace

    @classmethod
    def from_any(
        cls,
        idea_space_roots: Sequence[Path | str] | Path | str,
        workspace_root: Path | str | None = None,
        default_namespace: Namespace | None = None,
    ) -> Self:
        if isinstance(workspace_root, str):
            workspace_root = Path(workspace_root)
        if isinstance(idea_space_roots, (Path, str)):
            idea_space_roots = [idea_space_roots]
        roots = [Path(root).resolve() for root in idea_space_roots]
        idea_spaces = {
            root.name: IdeaSpace.from_path(path=root, root=root, with_creation=True)
            for root in roots
        }
        if len(idea_spaces) != len(roots):
            raise ValueError("IdeaSpace roots must have unique directory names.")

        if workspace_root is not None:
            workspace_root = Path(workspace_root).resolve()
            exclude_patterns = set(WorkspaceExplorer.DEFAULT_EXCLUDE_PATTERNS)
            for root in roots:
                if root.is_relative_to(workspace_root) and root != workspace_root:
                    relative_path = root.relative_to(workspace_root)
                    exclude_patterns.add(relative_path.as_posix())
            workspace_explorer = WorkspaceExplorer.from_any(
                workspace=workspace_root,
                excludes=exclude_patterns,
            )
        else:
            workspace_explorer = None
        return cls(
            idea_spaces=idea_spaces,
            workspace_explorer=workspace_explorer,
            default_namespace=default_namespace,
        )

    @property
    def workspace_root(self) -> Path | None:
        if self._workspace_explorer is not None:
            return self._workspace_explorer.workspace
        return None

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
            self.get_idea_space_convention,
            self.get_updated_time_of_idea_notes,
            self.create_sub_idea_space,
            self.delete_sub_idea_space,
            self.get_metadata_of_idea_notes,
            self.update_metadata_of_idea_note,
            self.get_idea_note,
            self.write_idea_note,
            self.delete_idea_note,
        ]
        if self._workspace_explorer is not None:
            tools = [*tools, *self._workspace_explorer.tools]
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
        if isinstance(tool_context_path := self.get_tool_context_path(idea_namespace), ToolError):
            return tool_context_path

        try:
            text = tool_context_path.read_text()
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

    def get_all_sub_idea_spaces_supported_by_convention(
        self,
        idea_namespace: Namespace | None = None,
    ) -> Sequence[IdeaSpacePath] | ToolError:
        """Direct-call compatibility alias for the discovery tool."""
        return self._discovery.get_all_sub_idea_spaces_supported_by_convention(idea_namespace)

    @tool_inspection_boundary
    def get_idea_space_convention(
        self,
        idea_space_path: IdeaSpacePath = ".",
        idea_namespace: Namespace | None = None,
    ) -> IdeaSpaceConventionModel | None | ToolError:
        """Get the convention defined directly at an IdeaSpace path.

        Parent IdeaSpaces are not searched for inherited conventions.

        Args:
            idea_space_path:
                IdeaSpace-root-relative path. ``.`` refers to the root.
            idea_namespace:
                Namespace of the IdeaSpace. ``null`` uses the default namespace.

        Returns:
            The convention defined at ``idea_space_path``, or ``null`` when none is
            defined. Returns ``ToolError`` when the path cannot be inspected.
        """

        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        resolved_namespace = self._resolve_idea_namespace(idea_namespace)
        if isinstance(resolved_namespace, ToolError):
            return resolved_namespace

        try:
            file_path = idea_space.resolve(idea_space_path)
            sub_idea_space = IdeaSpace.from_path(file_path, root=idea_space.root_directory_path)
            convention = sub_idea_space.convention
            if convention is not None:
                idea_note_model = build_idea_note_model(
                    convention,
                    self._idea_graphs[resolved_namespace],
                    workspace_root=self.workspace_root,
                    namespace=resolved_namespace,
                )
                if isinstance(idea_note_model, IdeaNoteModel):
                    return IdeaSpaceConventionModel(
                        idea_note=idea_note_model,
                        applied_to=sub_idea_space.idea_path,
                    )
                else:
                    return idea_note_model
            return None
        except ValueError as exc:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=str(exc))
        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc), retry=False)

    def get_sub_idea_spaces(
        self,
        idea_space_pivot: IdeaSpacePivot = ".",
        depth: IdeaSpaceDepth = 0,
        idea_namespace: Namespace | None = None,
    ) -> Sequence[IdeaSpacePath] | ToolError:
        """Direct-call compatibility alias for the discovery tool."""
        return self._discovery.get_sub_idea_spaces(
            idea_space_pivot=idea_space_pivot,
            depth=depth,
            idea_namespace=idea_namespace,
        )

    def create_sub_idea_space(
        self,
        idea_space_path: IdeaSpacePath,
        idea_namespace: Namespace | None = None,
    ) -> IdeaSpacePath | ToolError:
        """Create a new empty IdeaSpace directory.

        ``idea_space_path`` must identify a new directory below the IdeaSpace root. Its
        parent directory must already exist. Existing directories and reserved
        metadata directories are not treated as successful creation.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        try:
            directory_path = idea_space.resolve(idea_space_path)
            if directory_path == idea_space.root_directory_path:
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg="The IdeaSpace root already exists and cannot be created as a subspace.",
                )
            if directory_path.name == IdeaSpace.SPACE_META_NAME:
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{idea_space_path}` is reserved for IdeaSpace tool metadata.",
                )
            if directory_path.exists():
                if directory_path.is_dir():
                    msg = f"IdeaSpace already exists at `{idea_space_path}`."
                else:
                    msg = f"`{idea_space_path}` already exists and is not a directory."
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=msg,
                )
            if not directory_path.parent.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Parent of `{idea_space_path}` does not exist.",
                    suggestion=f"How about creating a subspace at `{Path(idea_space_path).parent.as_posix()}`",
                )
            directory_path.mkdir()
        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except FileNotFoundError as exc:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc), retry=False)
        return idea_space_path

    def delete_sub_idea_space(
        self,
        idea_space_path: IdeaSpacePath,
        idea_namespace: Namespace | None = None,
    ) -> IdeaSpacePath | ToolError:
        """Delete an existing empty IdeaSpace subdirectory.

        The IdeaSpace root, reserved metadata directories, and non-empty
        directories cannot be deleted by this operation.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        try:
            directory_path = idea_space.resolve(idea_space_path)
            if directory_path == idea_space.root_directory_path:
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg="The IdeaSpace root cannot be deleted.",
                )
            if directory_path.name == IdeaSpace.SPACE_META_NAME:
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{idea_space_path}` is reserved for IdeaSpace tool metadata.",
                )
            if not directory_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{idea_space_path}` does not identify an IdeaSpace directory.",
                )
            directory_path.rmdir()
        except FileNotFoundError as exc:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(exc), retry=False)
        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg=f"IdeaSpace `{idea_space_path}` must be empty before it can be deleted: {exc}",
                retry=False,
            )
        return idea_space_path

    def get_idea_note_paths(
        self,
        idea_space_pivot: IdeaSpacePivot = ".",
        depth: IdeaSpaceDepth = 0,
        idea_namespace: Namespace | None = None,
    ) -> Sequence[IdeaNotePath] | ToolError:
        """Direct-call compatibility alias for the discovery tool."""
        return self._discovery.get_idea_note_paths(
            idea_space_pivot=idea_space_pivot,
            depth=depth,
            idea_namespace=idea_namespace,
        )

    def get_updated_time_of_idea_notes(
        self,
        idea_space_pivot: IdeaSpacePivot = ".",
        depth: IdeaSpaceDepth = 0,
        idea_namespace: Namespace | None = None,
    ) -> dict[IdeaNotePath, AwareDatetime] | ToolError:
        """Get the file modification time of each IdeaNote under the specified IdeaSpace pivot.

        This is the filesystem modification time, not an LLM edit timestamp.
        A timestamp later than a previously recorded time indicates that the file
        was modified after that time, but this tool does not identify who made the
        change.

        Args:
            depth:
                Number of descendant levels to inspect. ``0`` inspects notes
                directly under ``pivot``. ``null`` inspects all descendants.

        Returns:
            A mapping from each IdeaSpace-relative note path to its last file
            modification time. Timestamps are timezone-aware UTC datetimes.

            ``ToolError`` if ``pivot`` is outside the IdeaSpace or the notes
            cannot be inspected.
        """

        def _get_file_modified_time(file_path: Path) -> AwareDatetime:
            return datetime.fromtimestamp(
                file_path.stat().st_mtime,
                tz=timezone.utc,
            )

        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        try:
            pivot_path = idea_space.resolve(idea_space_pivot)
            notes = IdeaSpace.from_path(pivot_path, root=idea_space.root_directory_path).get_notes(
                depth=depth
            )
            return {note.idea_path: _get_file_modified_time(note.file_path) for note in notes}

        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc), retry=False)

    @tool_discovery_boundary
    def get_metadata_of_idea_notes(
        self,
        idea_note_paths: Sequence[IdeaNotePath],
        idea_namespace: Namespace | None = None,
    ) -> dict[IdeaNotePath, IdeaNoteMetadata | None] | ToolError:
        """Get metadata for multiple IdeaNotes.

        For each input path, return a metadata object when the path identifies
        an IdeaNote. Return an empty object ``{}`` when the note has no
        metadata. Return `null` when the path is invalid, does not exist, identifies a directory,
        or the note's metadata cannot be deserialized.

        Return ``ToolError`` only when the metadata operation cannot be
        completed because of an I/O error. The returned mapping uses the
        requested paths as keys.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        metadata_by_path: dict[IdeaNotePath, IdeaNoteMetadata | None] = {}

        for path in idea_note_paths:
            try:
                file_path = idea_space.resolve(path)
                if file_path.is_dir():
                    metadata_by_path[path] = None
                    continue
                idea_note = IdeaNote.from_path(path=file_path, root=idea_space.root_directory_path)
            except (FileNotFoundError, MetadataDeserializationError):
                metadata_by_path[path] = None
            else:
                metadata_by_path[path] = idea_note.metadata.as_dict()

        return metadata_by_path

    def update_metadata_of_idea_note(
        self,
        idea_note_path: IdeaNotePath,
        idea_note_metadata: IdeaNoteMetadata,
        clear: bool = False,
        idea_namespace: Namespace | None = None,
    ) -> IdeaNotePath | ToolError:
        """Add or replace metadata fields of an existing IdeaNote.

        When ``clear`` is true, remove all existing metadata before applying
        ``metadata``. The Markdown body is always preserved.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        try:
            file_path = idea_space.resolve(idea_note_path)
            if file_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Given `{idea_note_path=}` corresponds to `IdeaSpace`, not a path to `IdeaNote`.",
                    suggestion="Use `get_idea_note_paths` to get the paths of `IdeaNote`.",
                )

            idea_note = IdeaNote.from_path(path=file_path, root=idea_space.root_directory_path)
            if clear:
                idea_note.metadata.clear()
            for key, value in idea_note_metadata.items():
                idea_note.metadata[key] = value
            idea_note.write(self._file_writer)

        except MetadataValueError:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="Given metadata is invalid.",
                retry=False,
            )

        except FileNotFoundError as exc:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(exc), retry=False)
        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc), retry=False)

        return idea_note_path

    @tool_inspection_boundary
    def get_idea_note(
        self,
        idea_note_path: IdeaNotePath,
        idea_namespace: Namespace | None = None,
    ) -> IdeaNoteModel | ToolError:
        """Read an IdeaNote and resolve its outgoing links.

        Args:
            idea_note_path:
                IdeaSpace-root-relative path of the note.
            idea_namespace:
                Namespace of the IdeaSpace. ``null`` uses the default namespace.

        Returns:
            The note model, or ``ToolError`` if the note cannot be read or parsed.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        resolved_namespace = self._resolve_idea_namespace(idea_namespace)
        if isinstance(resolved_namespace, ToolError):
            return resolved_namespace

        file_path = idea_space.resolve(idea_note_path)
        if file_path.is_dir():
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg=f"Given `{idea_note_path=}` corresponds to `IdeaSpacePath`, not a path to `IdeaNote` ",
                suggestion="Use `get_idea_note_paths` to get the paths of `IdeaNote`. ",
            )

        idea_note = IdeaNote.from_path(path=file_path, root=idea_space.root_directory_path)

        return build_idea_note_model(
            idea_note,
            self._idea_graphs[resolved_namespace],
            workspace_root=self.workspace_root,
            namespace=resolved_namespace,
        )

    def write_idea_note(
        self,
        idea_note_path: IdeaNotePath,
        idea_note_body: IdeaNoteBody,
        idea_note_metadata: IdeaNoteMetadata,
        idea_namespace: Namespace | None = None,
    ) -> IdeaNotePath | ToolError:
        """Create or replace an IdeaNote.

        Existing body and metadata are replaced rather than merged. ``idea_note_body``
        must not contain YAML frontmatter; metadata is supplied separately.

        Args:
            idea_note_path:
                IdeaSpace-root-relative path. ``.`` refers to the root.

            idea_namespace:
                Namespace of the IdeaSpace. ``null`` uses the default namespace.
        Returns:
            The written IdeaSpace-root-relative path, or ``ToolError`` on failure.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        try:
            file_path = idea_space.resolve(idea_note_path)
            if not file_path.parent.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg="Parent IdeaSpace does not exist.",
                    suggestion=f"Create the parent IdeaSpace first: `{Path(idea_note_path).parent.as_posix()}`.",
                    retry=False,
                )

            idea_note = IdeaNote.create(
                file_path=file_path, body=idea_note_body, root=idea_space.root_directory_path
            )
            for key, value in idea_note_metadata.items():
                idea_note.metadata[key] = value
            idea_note.write(self._file_writer)
        except OutsidePathError:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=(f"`{idea_note_path}` is outside of `IdeaSpace`."),
                retry=False,
            )
        except MetadataValueError:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg="Given metadata is invalid as the key and value.",
                retry=False,
            )
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc))
        return idea_note_path

    def delete_idea_note(
        self,
        idea_note_path: IdeaNotePath,
        idea_namespace: Namespace | None = None,
    ) -> IdeaNotePath | ToolError:
        """Permanently delete an existing IdeaNote.

        This operation cannot be undone. ``idea_note_path`` must identify a note, not a
        directory.

        Returns the IdeaSpace-root-relative path after a successful deletion,
        or ``ToolError`` if the note cannot be deleted.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        try:
            file_path = idea_space.resolve(idea_note_path)

            if file_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Given `{idea_note_path=}` corresponds to `IdeaSpace`, not a path to `IdeaNote`.",
                    suggestion="Use `get_idea_note_paths` to get the paths of `IdeaNote`.",
                )
            file_path.unlink()
        except FileNotFoundError as exc:
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND,
                msg=str(exc),
                retry=False,
            )
        except OutsidePathError as exc:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=str(exc),
                retry=False,
            )
        except OSError as exc:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR,
                msg=str(exc),
                retry=False,
            )

        return idea_note_path
