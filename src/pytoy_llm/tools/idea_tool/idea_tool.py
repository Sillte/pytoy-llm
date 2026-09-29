from datetime import datetime, timezone
from pathlib import Path
from posixpath import normpath
from typing import Callable, Self, Sequence
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
    IdeaSpaceDepth,
    IdeaSpacePath,
    IdeaSpacePivot,
)


def build_idea_note_model(
    idea_note: IdeaNote, idea_graph: IdeaGraph, workspace_root: Path | None
) -> IdeaNoteModel | ToolError:
    idea_space = idea_graph.space
    idea_links = idea_graph.resolve_links(idea_note)

    note_links = []
    remote_links = []
    local_links = []
    unresolved_links = []

    for idea_link in idea_links:
        try:
            if idea_link.uri.scheme == "idea":
                target_path = _relative_path_from_uri(idea_link.uri.path, idea_link.uri.authority)
                note_links.append(
                    IdeaNoteLinkModel(
                        source_idea_note_path=idea_note.idea_path,
                        target_idea_note_path=target_path,
                        idea_space_directory_folder=idea_space.root_directory_path,
                    )
                )
            elif idea_link.uri.scheme == "workspace":
                if workspace_root is not None:
                    workspace_path = _relative_path_from_uri(
                        idea_link.uri.path, idea_link.uri.authority
                    )
                    local_links.append(
                        WorkspaceLinkModel(
                            idea_note_path=idea_note.idea_path,
                            workspace_file_path=workspace_path,
                            workspace_root_folder=workspace_root,
                        )
                    )
                else:
                    raise ValueError("Workspace is not given here.")
            else:
                remote_links.append(
                    RemoteLinkModel(
                        idea_note_path=idea_note.idea_path,
                        uri=idea_link.uri,
                    )
                )
        except (ValueError, TypeError) as exc:
            unresolved_links.append(
                UnresolvedLinkModel(
                    idea_note_path=idea_note.idea_path,
                    uri=idea_link.uri,
                    reason=str(exc),
                )
            )

    try:
        modified_at = datetime.fromtimestamp(
            idea_note.file_path.stat().st_mtime,
            timezone.utc,
        )
    except OSError as exc:
        return ToolError(
            kind=ToolErrorKind.IO_ERROR,
            msg=str(exc),
            retry=False,
        )

    try:
        return IdeaNoteModel(
            idea_note_path=idea_note.idea_path,
            modified_at=modified_at,
            body=idea_note.body,
            metadata=idea_note.metadata.as_dict(),
            idea_note_links=note_links,
            remote_links=remote_links,
            workspace_local_links=local_links,
            unresolved_links=unresolved_links,
        )
    except ValueError as exc:
        return ToolError(
            kind=ToolErrorKind.INVALID_ARGUMENT,
            msg=str(exc),
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
        self, idea_space: IdeaSpace, workspace_explorer: WorkspaceExplorer | None = None
    ) -> None:
        self._idea_space = idea_space
        self._idea_graph = IdeaGraph(self._idea_space)
        self._file_writer = DiskFileWriter()
        self._workspace_explorer = workspace_explorer

        self._started_at: datetime = datetime.now(tz=timezone.utc)

    @classmethod
    def from_any(
        cls, idea_space_root: Path | str, workspace_root: Path | str | None = None
    ) -> Self:
        if isinstance(workspace_root, str):
            workspace_root = Path(workspace_root)

        idea_space_root = Path(idea_space_root).resolve()

        idea_space = IdeaSpace.from_path(
            path=idea_space_root,
            root=idea_space_root,
            with_creation=True,
        )

        if workspace_root is not None:
            workspace_root = Path(workspace_root).resolve()
            exclude_patterns = set(WorkspaceExplorer.DEFAULT_EXCLUDE_PATTERNS)
            if idea_space_root.is_relative_to(workspace_root) and idea_space_root != workspace_root:
                relative_path = idea_space_root.relative_to(workspace_root)
                exclude_patterns.add(relative_path.as_posix())
            workspace_explorer = WorkspaceExplorer.from_any(
                workspace=workspace_root,
                excludes=exclude_patterns,
            )
        else:
            workspace_explorer = None
        return cls(
            idea_space=idea_space,
            workspace_explorer=workspace_explorer,
        )

    @property
    def workspace_root(self) -> Path | None:
        if self._workspace_explorer is not None:
            return self._workspace_explorer.workspace
        return None

    @property
    def ideaspace_root(self) -> Path:
        return self._idea_space.directory_path

    @property
    def tool_context_path(self) -> Path:
        return self._idea_space.space_meta_directory / "tool_context.json"

    @property
    def tools(self) -> Sequence[Callable]:
        tools = [
            self.get_idea_space_working_context,
            self.get_all_sub_idea_spaces_supported_by_convention,
            self.get_idea_space_convention,
            self.get_updated_time_of_idea_notes,
            self.get_sub_idea_spaces,
            self.create_sub_idea_space,
            self.delete_sub_idea_space,
            self.get_idea_note_paths,
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
        self.tool_context_path.parent.mkdir(exist_ok=True, parents=True)
        self.tool_context_path.write_text(model.model_dump_json(indent=2), encoding="utf8")

    def get_idea_space_working_context(self) -> IdeaSpaceToolWorkingContextModel | ToolError:
        """Get the current tool-related working context for the IdeaSpace.

        The working context records the start and completion times of the latest
        LLM interaction. A missing context file is treated as an initialized
        context with both timestamps set to ``null``; it is not an error.

        Returns:
            ``IdeaSpaceToolWorkingContextModel`` containing the latest recorded
            interaction timestamps. Both timestamps are ``null`` when no interaction
            has been recorded.

            ``ToolError`` if the saved context exists but cannot be parsed.
        """
        try:
            text = self.tool_context_path.read_text()
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
    ) -> Sequence[IdeaSpacePath] | ToolError:
        """List all IdeaSpace paths where a convention is defined.

        The returned paths are relative to the IdeaSpace root. The root is
        represented by ``.``.

        Use each returned path as the ``idea_space_path`` argument of ``get_idea_space_convention`` to
        read the convention that applies to that path. This tool returns idea paths only;
        it does not return convention contents.

        Returns:
            A sequence of IdeaSpace-relative pivots sorted by path. An empty sequence
            means that no convention is defined in the IdeaSpace.

            ``ToolError`` if the IdeaSpace cannot be inspected.
        """
        try:
            spaces = [self._idea_space, *self._idea_space.get_subspaces(depth=None)]
            return sorted(space.idea_path for space in spaces if space.convention is not None)
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)

    def get_idea_space_convention(
        self, idea_space_path: IdeaSpacePath = "."
    ) -> IdeaSpaceConventionModel | None | ToolError:
        """Get the convention that applies to all notes and subspaces under the given IdeaSpace.

        This tool only checks the convention defined directly at ``idea_space_path``.
        It does not search parent paths for conventions.

        Returns ``null`` if no convention is defined directly at ``idea_space_path``.
        """

        try:
            file_path = self._idea_space.resolve(idea_space_path)
            idea_space = IdeaSpace.from_path(file_path, root=self.ideaspace_root)
            convention = idea_space.convention
            if convention is not None:
                idea_note_model = build_idea_note_model(
                    convention, IdeaGraph(idea_space), workspace_root=self.workspace_root
                )
                if isinstance(idea_note_model, IdeaNoteModel):
                    return IdeaSpaceConventionModel(
                        idea_note=idea_note_model,
                        applied_to=idea_space.idea_path,
                    )
                else:
                    return idea_note_model
            return None
        except ValueError as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=str(e))
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)

    def get_sub_idea_spaces(
        self, idea_space_pivot: IdeaSpacePivot = ".", depth: IdeaSpaceDepth = 0
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
        try:
            path = self._idea_space.resolve(idea_space_pivot)
            sub_space = IdeaSpace.from_path(path, root=self._idea_space.root_directory_path)
            result_spaces = sub_space.get_subspaces(depth=depth)
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)

        return [space.idea_path for space in result_spaces]

    def create_sub_idea_space(self, idea_space_path: IdeaSpacePath) -> IdeaSpacePath | ToolError:
        """Create a new empty IdeaSpace directory.

        ``idea_space_path`` must identify a new directory below the IdeaSpace root. Its
        parent directory must already exist. Existing directories and reserved
        metadata directories are not treated as successful creation.
        """
        try:
            directory_path = self._idea_space.resolve(idea_space_path)
            if directory_path == self.ideaspace_root:
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
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except FileNotFoundError as e:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)
        return idea_space_path

    def delete_sub_idea_space(self, idea_space_path: IdeaSpacePath) -> IdeaSpacePath | ToolError:
        """Delete an existing empty IdeaSpace subdirectory.

        The IdeaSpace root, reserved metadata directories, and non-empty
        directories cannot be deleted by this operation.
        """
        try:
            directory_path = self._idea_space.resolve(idea_space_path)
            if directory_path == self.ideaspace_root:
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
        except FileNotFoundError as e:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(e), retry=False)
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(
                kind=ToolErrorKind.INVALID_ARGUMENT,
                msg=f"IdeaSpace `{idea_space_path}` must be empty before it can be deleted: {e}",
                retry=False,
            )
        return idea_space_path

    def get_idea_note_paths(
        self, idea_space_pivot: IdeaSpacePivot = ".", depth: IdeaSpaceDepth = 0
    ) -> Sequence[IdeaNotePath] | ToolError:
        """List IdeaNote paths where are descendants of ``idea_space_pivot``.

        Args:
            depth:
                ``0`` returns only notes directly under ``pivot``. ``null``
                returns notes at all descendant levels.

        Returns:
            IdeaSpace-root-relative paths of matching IdeaNotes. This tool
            returns paths only.

            ``ToolError`` if ``idea_space_pivot`` is outside the IdeaSpace or cannot be
            inspected.
        """
        try:
            path = self._idea_space.resolve(idea_space_pivot)
            sub_space = IdeaSpace.from_path(path, root=self._idea_space.root_directory_path)
            result_notes = sub_space.get_notes(depth=depth)
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)
        return [note.idea_path for note in result_notes]

    def get_updated_time_of_idea_notes(
        self,
        idea_space_pivot: IdeaSpacePivot = ".",
        depth: IdeaSpaceDepth = 0,
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

        def _path_to_aware_datetime(path: Path) -> AwareDatetime:
            return datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            )

        try:
            path = self._idea_space.resolve(idea_space_pivot)
            sub_space = IdeaSpace.from_path(path, root=self._idea_space.root_directory_path)
            notes = sub_space.get_notes(depth=depth)
            return {note.idea_path: _path_to_aware_datetime(note.file_path) for note in notes}

        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)

    def get_metadata_of_idea_notes(
        self, idea_note_paths: Sequence[IdeaNotePath]
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
        metadata_by_path: dict[IdeaNotePath, IdeaNoteMetadata | None] = {}

        for path in idea_note_paths:
            try:
                file_path = self._idea_space.resolve(path)
                if file_path.is_dir():
                    metadata_by_path[path] = None
                    continue

                idea_note = IdeaNote.from_path(
                    path=file_path, root=self._idea_space.root_directory_path
                )
            except (FileNotFoundError, OutsidePathError, MetadataDeserializationError):
                metadata_by_path[path] = None
            except OSError as e:
                return ToolError(
                    kind=ToolErrorKind.IO_ERROR,
                    msg=str(e),
                    retry=False,
                    suggestion=f"Exclude `{path=}` from paths.",
                )
            else:
                metadata_by_path[path] = idea_note.metadata.as_dict()

        return metadata_by_path

    def update_metadata_of_idea_note(
        self,
        idea_note_path: IdeaNotePath,
        idea_note_metadata: IdeaNoteMetadata,
        clear: bool = False,
    ) -> IdeaNotePath | ToolError:
        """Add or replace metadata fields of an existing IdeaNote.

        When ``clear`` is true, remove all existing metadata before applying
        ``metadata``. The Markdown body is always preserved.
        """
        try:
            file_path = self._idea_space.resolve(idea_note_path)
            if file_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Given `{idea_note_path=}` corresponds to `IdeaSpace`, not a path to `IdeaNote`.",
                    suggestion="Use `get_idea_note_paths` to get the paths of `IdeaNote`.",
                )

            idea_note = IdeaNote.from_path(
                path=file_path, root=self._idea_space.root_directory_path
            )
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

        except FileNotFoundError as e:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(e), retry=False)
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)

        return idea_note_path

    def get_idea_note(self, idea_note_path: IdeaNotePath) -> IdeaNoteModel | ToolError:
        """Read one IdeaNote, including its body, metadata, and outgoing links.

        The returned ``body`` excludes YAML frontmatter. Links are separated
        into links to other IdeaNotes, workspace-local files, remote URIs, and
        unresolved links.

        """
        try:
            file_path = self._idea_space.resolve(idea_note_path)
            if file_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Given `{idea_note_path=}` corresponds to `IdeaSpacePath`, not a path to `IdeaNote` ",
                    suggestion="Use `get_idea_note_paths` to get the paths of `IdeaNote`. ",
                )

            idea_note = IdeaNote.from_path(
                path=file_path, root=self._idea_space.root_directory_path
            )
        except FileNotFoundError as e:
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND,
                msg=str(e),
                retry=False,
            )
        except OutsidePathError as e:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=str(e),
                retry=False,
            )
        except OSError as e:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR,
                msg=str(e),
                retry=False,
            )
        except MetadataDeserializationError:
            return ToolError(
                kind=ToolErrorKind.PARSE_ERROR,
                msg="The specified IdeaNote is broken.",
                retry=False,
            )

        return build_idea_note_model(
            idea_note, IdeaGraph(self._idea_space), workspace_root=self.workspace_root
        )

    def write_idea_note(
        self,
        idea_note_path: IdeaNotePath,
        idea_note_body: IdeaNoteBody,
        idea_note_metadata: IdeaNoteMetadata,
    ) -> IdeaNotePath | ToolError:
        """Create or replace an IdeaNote with the given content.

        If a note already exists, its body and metadata are replaced entirely.
        This operation does not append to or merge with the existing note.
        Provide Markdown without YAML frontmatter in ``body`` and provide
        frontmatter values through ``idea_note_metadata``.

        Returns the IdeaSpace-root-relative path after a successful write, or
        ``ToolError`` if the note cannot be written.
        """
        try:
            file_path = self._idea_space.resolve(idea_note_path)
            if not file_path.parent.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg="Parent IdeaSpace does not exist.",
                    suggestion=f"Create the parent IdeaSpace first: `{Path(idea_note_path).parent.as_posix()}`.",
                    retry=False,
                )

            idea_note = IdeaNote.create(
                file_path=file_path, body=idea_note_body, root=self._idea_space.root_directory_path
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
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e))
        return idea_note_path

    def delete_idea_note(self, idea_note_path: IdeaNotePath) -> IdeaNotePath | ToolError:
        """Permanently delete an existing IdeaNote.

        This operation cannot be undone. ``idea_note_path`` must identify a note, not a
        directory.

        Returns the IdeaSpace-root-relative path after a successful deletion,
        or ``ToolError`` if the note cannot be deleted.
        """
        try:
            file_path = self._idea_space.resolve(idea_note_path)

            if file_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Given `{idea_note_path=}` corresponds to `IdeaSpace`, not a path to `IdeaNote`.",
                    suggestion="Use `get_idea_note_paths` to get the paths of `IdeaNote`.",
                )
            file_path.unlink()
        except FileNotFoundError as e:
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND,
                msg=str(e),
                retry=False,
            )
        except OutsidePathError as e:
            return ToolError(
                kind=ToolErrorKind.PERMISSION_DENIED,
                msg=str(e),
                retry=False,
            )
        except OSError as e:
            return ToolError(
                kind=ToolErrorKind.IO_ERROR,
                msg=str(e),
                retry=False,
            )

        return idea_note_path
