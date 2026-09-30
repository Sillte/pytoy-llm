from pathlib import Path
from typing import Callable, Sequence

from pytoy_llm.idea import (
    DiskFileWriter,
    FileWriterProtocol,
    IdeaNote,
    IdeaSpace,
    MetadataValueError,
    OutsidePathError,
)
from pytoy_llm.tools.errors import ToolError, ToolErrorKind

from .semantic_types import IdeaNoteBody, IdeaNoteMetadata, IdeaNotePath, IdeaSpacePath, Namespace


class IdeaMutation:
    """Provide tools for creating, updating, and deleting IdeaNotes and IdeaSpaces."""

    def __init__(
        self,
        get_idea_space: Callable[[Namespace | None], IdeaSpace | ToolError],
        file_writer: FileWriterProtocol | None = None,
    ) -> None:
        self._get_idea_space = get_idea_space
        self._file_writer = file_writer or DiskFileWriter()

    @property
    def tools(self) -> Sequence[Callable]:
        return [
            self.create_sub_idea_space,
            self.delete_sub_idea_space,
            self.update_metadata_of_idea_note,
            self.write_idea_note,
            self.delete_idea_note,
        ]

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
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(exc), retry=False)
        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc), retry=False)

        return idea_note_path
