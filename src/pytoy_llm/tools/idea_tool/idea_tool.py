from pathlib import Path
from typing import Callable, Sequence

from pytoy_llm.idea import DiskFileWriter, IdeaGraph, IdeaNote, IdeaSpace
from pytoy_llm.tools.errors import ToolError, ToolErrorKind

from .models import IdeaNoteLinkModel, IdeaNoteModel, LocalLinkModel, RemoteLinkModel
from .semantic_types import IdeaNoteBody, IdeaNoteMetadata, IdeaSpaceDepth, IdeaSpacePath


class IdeaTool:
    def __init__(self, idea_space: IdeaSpace) -> None:
        self._idea_space = idea_space
        self._idea_graph = IdeaGraph(self._idea_space)
        self._file_writer = DiskFileWriter()

    @property
    def workspace_root(self) -> Path:
        return self._idea_space.root

    @property
    def ideaspace_root(self) -> Path:
        return self._idea_space.folder_path

    @property
    def tools(self) -> Sequence[Callable]:
        return [
            self.get_subspaces,
            self.get_note_paths,
            self.get_note,
            self.write_note,
        ]

    def get_subspaces(
        self, pivot: IdeaSpacePath = "./", depth: IdeaSpaceDepth = 0
    ) -> Sequence[IdeaSpacePath] | ToolError:
        """Get subspaces under a `pivot` in the IdeaSpace."""
        path = self._idea_space.folder_path / pivot

        try:
            sub_space = IdeaSpace.from_path(path, root=self.workspace_root)
            result_spaces = sub_space.get_subspaces(depth=depth)
        except PermissionError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)

        return [
            (space.folder_path.relative_to(self.ideaspace_root).as_posix())
            for space in result_spaces
        ]

    def get_note_paths(
        self, pivot: IdeaSpacePath = "./", depth: IdeaSpaceDepth = 0
    ) -> Sequence[IdeaSpacePath] | ToolError:
        """Get path of `IdeaNote` under a `pivot` in the IdeaSpace."""
        path = self._idea_space.folder_path / pivot

        try:
            sub_space = IdeaSpace.from_path(path, root=self.workspace_root)
            result_notes = sub_space.get_notes(depth=depth)
        except PermissionError as e:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(e), retry=False)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e), retry=False)
        return [
            (space.file_path.relative_to(self.ideaspace_root).as_posix()) for space in result_notes
        ]

    def get_note(self, path: IdeaSpacePath) -> IdeaNoteModel | ToolError:
        """Get an IdeaNote and its links by IdeaSpacePath."""
        file_path = self._idea_space.folder_path / path
        try:
            idea_note = IdeaNote.from_path(
                file_path=file_path,
                root=self.workspace_root,
            )
        except FileNotFoundError as e:
            return ToolError(
                kind=ToolErrorKind.NOT_FOUND,
                msg=str(e),
                retry=False,
            )
        except PermissionError as e:
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
        idea_links = self._idea_graph.resolve_links(idea_note)

        note_links = []
        remote_links = []
        local_links = []

        for idea_link in idea_links:
            target_file_path = idea_link.target_file_path
            if target_file_path is not None:
                if target_file_path.is_relative_to(self.ideaspace_root):
                    target_path = target_file_path.relative_to(self.ideaspace_root).as_posix()
                    note_links.append(IdeaNoteLinkModel(path=path, target_path=target_path))
                else:
                    reference_path = target_file_path.relative_to(self.workspace_root).as_posix()
                    local_links.append(LocalLinkModel(path=path, reference_path=reference_path))
            else:
                remote_links.append(RemoteLinkModel(path=path, uri=idea_link.uri))

        try:
            note_model = IdeaNoteModel(
                path=path,
                body=idea_note.body,
                metadata=dict(idea_note.metadata),
                note_links=note_links,
                remote_links=remote_links,
                local_links=local_links,
            )
        except ValueError as e:
            return ToolError(kind=ToolErrorKind.INVALID_ARGUMENT, msg=str(e))

        return note_model

    def write_note(
        self,
        path: IdeaSpacePath,
        body: IdeaNoteBody,
        metadata: IdeaNoteMetadata,
    ) -> IdeaSpacePath | ToolError:
        """Create or replace an IdeaNote with the given content."""
        file_path = self.ideaspace_root / path
        idea_note = IdeaNote.create(file_path=file_path, body=body, root=self.workspace_root)
        for key, value in metadata.items():
            idea_note.metadata[key] = value
        try:
            idea_note.write(self._file_writer)
        except OSError as e:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(e))
        return path
