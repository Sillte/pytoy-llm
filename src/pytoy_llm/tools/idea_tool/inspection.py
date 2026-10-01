from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import unquote

from pydantic import AwareDatetime

from pytoy_llm.idea import (
    IdeaGraph,
    IdeaNote,
    IdeaSpace,
    MetadataDeserializationError,
    OutsidePathError,
    UriLocalPathResolver,
)
from pytoy_llm.tools.errors import ToolError, ToolErrorKind

from .boundaries import tool_inspection_boundary
from .models import (
    IdeaNoteLinkModel,
    IdeaNoteModel,
    IdeaSpaceConventionModel,
    LocalLinkModel,
    RemoteLinkModel,
    UnresolvedLinkModel,
)
from .semantic_types import (
    IdeaNoteMetadata,
    IdeaNotePath,
    IdeaNoteReference,
    IdeaSpaceDepth,
    IdeaSpacePath,
    IdeaSpacePivot,
    Namespace,
)


def build_idea_note_model(
    source_idea_note: IdeaNote,
    idea_graph: IdeaGraph,
    local_path_resolver: UriLocalPathResolver,
    namespace: Namespace,
) -> IdeaNoteModel | ToolError:
    idea_links = idea_graph.resolve_links(source_idea_note)

    idea_note_links = []
    local_links = []
    remote_links = []
    unresolved_links = []

    source_idea_note_reference = IdeaNoteReference(
        idea_note_path=source_idea_note.idea_path, namespace=namespace
    )

    for idea_link in idea_links:
        try:
            scheme, authority = idea_link.uri.scheme, idea_link.uri.authority
            if local_path_resolver.is_registered(scheme, authority):
                _ = local_path_resolver.resolve(idea_link.uri, source_idea_note.file_path.parent)
                if scheme == "idea":
                    authority = authority or namespace
                    idea_note_links.append(
                        IdeaNoteLinkModel(
                            source_idea_note_reference=source_idea_note_reference,
                            target_idea_note_reference=IdeaNoteReference(
                                idea_note_path=idea_link.uri.path.strip("/"),
                                namespace=authority,
                            ),
                        )
                    )
                else:
                    local_links.append(
                        LocalLinkModel(
                            source_idea_note_reference=source_idea_note_reference,
                            scheme=idea_link.uri.scheme,
                            namespace=idea_link.uri.authority,
                            path=unquote(idea_link.uri.path).strip("/"),
                        ),
                    )
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
        source_idea_note.file_path.stat().st_mtime,
        timezone.utc,
    )
    return IdeaNoteModel(
        idea_note_path=source_idea_note.idea_path,
        modified_at=modified_at,
        body=source_idea_note.body,
        metadata=source_idea_note.metadata.as_dict(),
        idea_note_links=idea_note_links,
        remote_links=remote_links,
        local_links=local_links,
        unresolved_links=unresolved_links,
    )


class IdeaInspection:
    """Provide read-only inspection tools for an IdeaSpace."""

    def __init__(
        self,
        get_idea_space: Callable[[Namespace | None], IdeaSpace | ToolError],
        resolve_idea_namespace: Callable[[Namespace | None], Namespace | ToolError],
        idea_graphs: Mapping[Namespace, IdeaGraph],
        local_path_resolver: UriLocalPathResolver,
    ) -> None:
        self._get_idea_space = get_idea_space
        self._resolve_idea_namespace = resolve_idea_namespace
        self._idea_graphs = idea_graphs
        self._local_path_resolver = local_path_resolver

    @property
    def tools(self) -> Sequence[Callable]:
        return [
            self.get_idea_space_convention,
            self.get_effective_conventions,
            self.get_updated_time_of_idea_notes,
            self.get_metadata_of_idea_notes,
            self.get_idea_note,
        ]

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
                    local_path_resolver=self._local_path_resolver,
                    namespace=resolved_namespace,
                )
                if isinstance(idea_note_model, IdeaNoteModel):
                    return IdeaSpaceConventionModel(
                        idea_note=idea_note_model,
                    )
                return idea_note_model
            return None
        except ValueError as exc:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=str(exc))
        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc), retry=False)

    @tool_inspection_boundary
    def get_effective_conventions(
        self,
        idea_space_or_note_path: IdeaSpacePath | IdeaNotePath,
        idea_namespace: Namespace | None = None,
    ) -> Mapping[IdeaSpacePath, IdeaSpaceConventionModel] | ToolError:
        """Get conventions effective for an IdeaNote or direct notes in an IdeaSpace.

        Paths ending in `.md` are interpreted as IdeaNote paths; other paths are
        interpreted as IdeaSpace paths. The returned mapping is ordered from the
        root IdeaSpace toward the target's containing or identified IdeaSpace.
        Each key is the IdeaSpace path where that convention is defined. Deeper
        conventions take precedence over shallower ones.

        Returns an empty mapping when no applicable convention is defined.
        Returns ``ToolError`` when the input path does not exist or is invalid.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        resolved_namespace = self._resolve_idea_namespace(idea_namespace)
        if isinstance(resolved_namespace, ToolError):
            return resolved_namespace

        input_path = Path(idea_space_or_note_path)
        target_path = idea_space.resolve(input_path)

        if input_path.name.endswith(".md"):
            if target_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Expected an IdeaNote path, but found a directory: {input_path}",
                )
            if not target_path.is_file():
                return ToolError(
                    kind=ToolErrorKind.NOT_FOUND,
                    msg=f"{input_path=} is not found.",
                )
            target_space_path = target_path.parent
        else:
            if not target_path.exists():
                return ToolError(
                    kind=ToolErrorKind.NOT_FOUND,
                    msg=f"{input_path=} is not found.",
                )
            target_space_path = target_path

        target_space = IdeaSpace.from_path(
            target_space_path,
            root=idea_space.root_directory_path,
        )
        ancestor_spaces = [target_space]
        while ancestor_spaces[-1].idea_path != ".":
            ancestor_spaces.append(ancestor_spaces[-1].parent)

        conventions: dict[IdeaSpacePath, IdeaSpaceConventionModel] = {}
        for ancestor_space in reversed(ancestor_spaces):
            convention = ancestor_space.convention
            if convention is None:
                continue

            idea_note_model = build_idea_note_model(
                convention,
                self._idea_graphs[resolved_namespace],
                local_path_resolver=self._local_path_resolver,
                namespace=resolved_namespace,
            )
            if isinstance(idea_note_model, ToolError):
                return idea_note_model
            conventions[ancestor_space.idea_path] = IdeaSpaceConventionModel(
                idea_note=idea_note_model,
            )

        return conventions

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

        def get_file_modified_time(file_path: Path) -> AwareDatetime:
            return datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)

        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        try:
            pivot_path = idea_space.resolve(idea_space_pivot)
            notes = IdeaSpace.from_path(pivot_path, root=idea_space.root_directory_path).get_notes(
                depth=depth
            )
            return {note.idea_path: get_file_modified_time(note.file_path) for note in notes}
        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc), retry=False)

    @tool_inspection_boundary
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
            local_path_resolver=self._local_path_resolver,
            namespace=resolved_namespace,
        )
