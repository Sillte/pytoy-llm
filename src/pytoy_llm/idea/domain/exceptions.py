class OutsidePathError(PermissionError):
    """Path is outside of `IdeaSpace`."""


class ShouldBeNotePathError(ValueError):
    """Path should be an `IdeaNote` path."""


class ShouldBeSpacePathError(ValueError):
    """Path should be an `IdeaSpace` path."""


class MetadataError(Exception):
    """Base exception for metadata operations."""


class MetadataKeyError(MetadataError, KeyError):
    """A requested metadata key does not exist."""


class MetadataValueError(MetadataError, ValueError):
    """A metadata key or value cannot be represented."""


class MetadataSerializationError(MetadataError, ValueError):
    """Metadata cannot be serialized."""


class MetadataDeserializationError(MetadataError, ValueError):
    """Metadata cannot be deserialized."""
