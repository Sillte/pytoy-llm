from typing import Any, Iterable, Iterator, Protocol

from pydantic import JsonValue


class MetaDataProtocol(Protocol):
    """Mutable mapping-like metadata used by :class:`IdeaNote`.

    Raises:
        MetadataError: If a metadata operation cannot be completed. Concrete
            implementations should raise the most specific subclass available.
    """

    def __getitem__(self, key: str) -> Any:
        """Return the value for ``key``.

        Raises:
            MetadataKeyError: If ``key`` is not present.
        """
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        """Set ``key`` to ``value``.

        Raises:
            MetadataValueError: If the key or value cannot be represented.
        """
        ...

    def __delitem__(self, key: str) -> None:
        """Delete ``key``.

        Raises:
            MetadataKeyError: If ``key`` is not present.
        """
        ...

    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...

    def keys(self) -> Iterable[str]: ...

    def as_text(self) -> str:
        """Serialize metadata to text.

        Raises:
            MetadataSerializationError: If metadata cannot be serialized.
        """
        ...

    def as_dict(self) -> dict[str, JsonValue]:
        """Return metadata as JSON-compatible values.

        Raises:
            MetadataSerializationError: If metadata cannot be serialized.
        """
        ...

    def clear(self) -> None: ...
