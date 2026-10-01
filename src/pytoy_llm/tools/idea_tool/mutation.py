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
            self.move_idea_space,
            self.update_metadata_of_idea_note,
            self.write_idea_note,
            self.move_idea_note,
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

    def move_idea_space(
        self,
        source_idea_space_path: IdeaSpacePath,
        destination_idea_space_path: IdeaSpacePath,
        idea_namespace: Namespace | None = None,
    ) -> IdeaSpacePath | ToolError:
        """Move an IdeaSpace directory and all its contents within one namespace.

        The IdeaSpace root and reserved metadata directories cannot be moved.
        The destination parent must already exist, and an existing destination
        is never overwritten. Moving a space does not rewrite links, and its
        effective conventions may change under the new hierarchy.

        Returns the destination IdeaSpace-root-relative path, or ``ToolError``
        if the move cannot be completed. Moving a space to its current path is
        a successful no-op.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        try:
            source_path = idea_space.resolve(source_idea_space_path)
            destination_path = idea_space.resolve(destination_idea_space_path)
            root_path = idea_space.root_directory_path

            if Path(source_idea_space_path).suffix == ".md":
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{source_idea_space_path}` identifies an IdeaNote path, not an IdeaSpace.",
                )
            if Path(destination_idea_space_path).suffix == ".md":
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{destination_idea_space_path}` identifies an IdeaNote path, not an IdeaSpace.",
                )
            if source_path == root_path:
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg="The IdeaSpace root cannot be moved.",
                )
            if (
                IdeaSpace.SPACE_META_NAME in source_path.relative_to(root_path).parts
                or IdeaSpace.SPACE_META_NAME in destination_path.relative_to(root_path).parts
            ):
                return ToolError(
                    kind=ToolErrorKind.PERMISSION_DENIED,
                    msg="IdeaSpace tool metadata directories cannot be moved.",
                    retry=False,
                )
            if not source_path.exists():
                return ToolError(
                    kind=ToolErrorKind.NOT_FOUND,
                    msg=f"IdeaSpace `{source_idea_space_path}` does not exist.",
                    retry=False,
                )
            if not source_path.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{source_idea_space_path}` does not identify an IdeaSpace directory.",
                )
            if source_path == destination_path:
                return destination_idea_space_path
            if destination_path.is_relative_to(source_path):
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg="An IdeaSpace cannot be moved into itself or one of its descendants.",
                )
            if destination_path.exists():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Destination `{destination_idea_space_path}` already exists.",
                )
            if not destination_path.parent.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Parent IdeaSpace of `{destination_idea_space_path}` does not exist.",
                )

            source_path.rename(destination_path)
        except FileNotFoundError as exc:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(exc), retry=False)
        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc), retry=False)

        return destination_idea_space_path

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

    def move_idea_note(
        self,
        source_idea_note_path: IdeaNotePath,
        destination_idea_note_path: IdeaNotePath,
        idea_namespace: Namespace | None = None,
    ) -> IdeaNotePath | ToolError:
        """Move an IdeaNote to another path in the same IdeaSpace.

        Both paths must identify Markdown notes. The destination parent must
        already exist, and an existing destination is never overwritten.
        Moving a note does not rewrite links to or from it. Moving a
        ``.convention.md`` note changes the IdeaSpace where its convention
        applies.

        Returns the destination IdeaSpace-root-relative path, or ``ToolError``
        if the move cannot be completed. Moving a note to its current path is
        a successful no-op.
        """
        idea_space = self._get_idea_space(idea_namespace)
        if isinstance(idea_space, ToolError):
            return idea_space
        try:
            source_path = idea_space.resolve(source_idea_note_path)
            destination_path = idea_space.resolve(destination_idea_note_path)

            if source_path.suffix != ".md":
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{source_idea_note_path}` does not identify an IdeaNote path.",
                )
            if destination_path.suffix != ".md":
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{destination_idea_note_path}` does not identify an IdeaNote path.",
                )
            if not source_path.exists():
                return ToolError(
                    kind=ToolErrorKind.NOT_FOUND,
                    msg=f"IdeaNote `{source_idea_note_path}` does not exist.",
                    retry=False,
                )
            if not source_path.is_file():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"`{source_idea_note_path}` does not identify an IdeaNote file.",
                )
            if source_path == destination_path:
                return destination_idea_note_path
            if destination_path.exists():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Destination `{destination_idea_note_path}` already exists.",
                )
            if not destination_path.parent.is_dir():
                return ToolError(
                    kind=ToolErrorKind.INVALID_ARGUMENT,
                    msg=f"Parent IdeaSpace of `{destination_idea_note_path}` does not exist.",
                )

            source_path.rename(destination_path)
        except FileNotFoundError as exc:
            return ToolError(kind=ToolErrorKind.NOT_FOUND, msg=str(exc), retry=False)
        except OutsidePathError as exc:
            return ToolError(kind=ToolErrorKind.PERMISSION_DENIED, msg=str(exc), retry=False)
        except OSError as exc:
            return ToolError(kind=ToolErrorKind.IO_ERROR, msg=str(exc), retry=False)

        return destination_idea_note_path
