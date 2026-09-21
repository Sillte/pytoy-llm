from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Self, Sequence

from pytoy_llm.idea import DiskFileWriter, IdeaGraph, IdeaNote, IdeaSpace, OutsidePathError
from pytoy_llm.tools.errors import ToolError, ToolErrorKind
from pytoy_llm.tools.workspace_explorer import WorkspaceExplorer

from .models import (
    IdeaNoteLinkModel,
    IdeaNoteModel,
    IdeaSpaceContextModel,
    IdeaSpaceConventionModel,
    IdeaSpaceToolMetaModel,
    LocalLinkModel,
    RemoteLinkModel,
    UnresolvedLinkModel,
)
from .semantic_types import (
    IdeaNoteBody,
    IdeaNoteMetadata,
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
        target_file_path = idea_link.target_file_path
        try:
            if target_file_path is not None:
                if target_file_path.is_relative_to(idea_space.root):
                    target_path = target_file_path.relative_to(idea_space.root).as_posix()
                    note_links.append(
                        IdeaNoteLinkModel(path=idea_note.path, target_path=target_path)
                    )
                else:
                    if workspace_root is not None:
                        reference_path = target_file_path.relative_to(workspace_root).as_posix()
                        local_links.append(
                            LocalLinkModel(path=idea_note.path, reference_path=reference_path)
                        )
                    else:
                        raise ValueError("Workspace is not given here.")
            else:
                remote_links.append(RemoteLinkModel(path=idea_note.path, uri=idea_link.uri))
        except (ValueError, TypeError) as exc:
            unresolved_links.append(
                UnresolvedLinkModel(path=idea_note.path, uri=idea_link.uri, reason=str(exc))
            )

    try:
        note_model = IdeaNoteModel(
            path=idea_note.path,
            modified_at=datetime.fromtimestamp(idea_note.file_path.stat().st_mtime, timezone.utc),
            body=idea_note.body,
            metadata=dict(idea_note.metadata),
            note_links=note_links,
            remote_links=remote_links,
            local_links=local_links,
            unresolved_links=unresolved_links,
        )
    except ValueError as e:
        return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=str(e))

    return note_model


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

        idea_space = IdeaSpace.from_path(path=idea_space_root, root=idea_space_root)

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
        return self._idea_space.folder_path

    @property
    def tool_context_path(self) -> Path:
        return self._idea_space.space_meta_folder / "tool_context.json"

    @property
    def tools(self) -> Sequence[Callable]:
        tools = [
            self.get_idea_space_root_context,
            self.get_convention,
            self.get_subspaces,
            self.get_note_paths,
            self.get_note,
            self.write_note,
            self.delete_note,
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
        self.tool_context_path.write_text(model.model_dump_json(indent=2))

    def get_idea_space_root_context(self) -> IdeaSpaceContextModel | ToolError:
        convention = self.get_convention(pivot=".")
        if isinstance(convention, ToolError):
            return convention
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
        return IdeaSpaceContextModel(root_convention=convention, tool_meta=tool_meta)

    def get_convention(
        self, pivot: IdeaSpacePivot = "."
    ) -> IdeaSpaceConventionModel | None | ToolError:
        """Get the convention that applies at the given IdeaSpace pivot.
        Return `null` if no convention exists for the given pivot.

        When you first enter an IdeaSpace, before exploring or modifying it,
        first call `get_convention(".")` to learn the conventions specific to
        the IdeaSpace root. Then follow those conventions when navigating,
        creating, or modifying notes.
        """

        try:
            file_path = self._idea_space.resolve(pivot)
            idea_space = IdeaSpace.from_path(file_path, root=self.ideaspace_root)
            convention = idea_space.convention
            if convention is not None:
                idea_note_model = build_idea_note_model(
                    convention, IdeaGraph(idea_space), workspace_root=self.workspace_root
                )
                if isinstance(idea_note_model, IdeaNoteModel):
                    return IdeaSpaceConventionModel(note=idea_note_model, applied_to=pivot)
                else:
                    return idea_note_model
            return None
        except ValueError as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=str(e))
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)

    def get_subspaces(
        self, pivot: IdeaSpacePivot = ".", depth: IdeaSpaceDepth = 0
    ) -> Sequence[IdeaSpacePath] | ToolError:
        """Get subspaces under a `pivot` in the IdeaSpace."""
        try:
            path = self._idea_space.resolve(pivot)
            sub_space = IdeaSpace.from_path(path, root=self._idea_space.root)
            result_spaces = sub_space.get_subspaces(depth=depth)
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)

        return [
            (space.folder_path.relative_to(self.ideaspace_root).as_posix())
            for space in result_spaces
        ]

    def get_note_paths(
        self, pivot: IdeaSpacePivot = "./", depth: IdeaSpaceDepth = 0
    ) -> Sequence[IdeaSpacePath] | ToolError:
        """Get path of `IdeaNote` under a `pivot` in the IdeaSpace."""
        try:
            path = self._idea_space.resolve(pivot)
            sub_space = IdeaSpace.from_path(path, root=self._idea_space.root)
            result_notes = sub_space.get_notes(depth=depth)
        except OutsidePathError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)
        return [
            (note.file_path.relative_to(self.ideaspace_root).as_posix()) for note in result_notes
        ]

    def get_note(self, path: IdeaSpacePath) -> IdeaNoteModel | ToolError:
        """Get an IdeaNote and its links by IdeaSpacePath."""
        try:
            file_path = self._idea_space.resolve(path)
            if file_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Given `{path=}` corresponds to `IdeaSpacePivot`, not a path to `IdeaNote` ",
                    suggestion="Use `get_note_paths` to get the paths of `IdeaSpaceNote`. ",
                )

            idea_note = IdeaNote.from_path(file_path=file_path, root=self._idea_space.root)
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

        return build_idea_note_model(
            idea_note, IdeaGraph(self._idea_space), workspace_root=self.workspace_root
        )

    def write_note(
        self,
        path: IdeaSpacePath,
        body: IdeaNoteBody,
        metadata: IdeaNoteMetadata,
    ) -> IdeaSpacePath | ToolError:
        """Create or replace an IdeaNote with the given content."""
        try:
            file_path = self._idea_space.resolve(path)
            idea_note = IdeaNote.create(file_path=file_path, body=body, root=self._idea_space.root)
            for key, value in metadata.items():
                idea_note.metadata[key] = value
            idea_note.write(self._file_writer)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e))
        return path

    def delete_note(self, path: IdeaSpacePath) -> IdeaSpacePath | ToolError:
        """Delete an IdeaNote from the IdeaSpace."""
        try:
            file_path = self._idea_space.resolve(path)

            if file_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Given `{path=}` corresponds to `IdeaSpacePivot`, not a path to `IdeaNote`.",
                    suggestion="Use `get_note_paths` to get the paths of `IdeaSpaceNote`.",
                )

            idea_note = IdeaNote.from_path(
                file_path=file_path,
                root=self._idea_space.root,
            )
            idea_note.file_path.unlink()

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

        return path
