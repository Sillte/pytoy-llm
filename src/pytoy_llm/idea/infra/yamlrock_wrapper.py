from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Self, cast

import yamlrocks
from pydantic import JsonValue
from yamlrocks import YAMLRocksDocument

from pytoy_llm.idea.domain.exceptions import (
    MetadataDeserializationError,
    MetadataKeyError,
    MetadataSerializationError,
    MetadataValueError,
)


class YamlRockWrapper:
    def __init__(self, yaml_rock: YAMLRocksDocument | None = None) -> None:
        yaml_rock = yaml_rock or cast(
            YAMLRocksDocument, yamlrocks.loads("{}", option=yamlrocks.OPT_ROUND_TRIP)
        )
        self._yaml_rock = yaml_rock

    @classmethod
    def from_yaml_text(cls, yaml_text: str) -> Self:
        try:
            yaml_rock = cast(
                YAMLRocksDocument,
                yamlrocks.loads(yaml_text, option=yamlrocks.OPT_ROUND_TRIP),
            )
        except Exception as exc:
            raise MetadataDeserializationError("Metadata YAML could not be parsed.") from exc
        return cls(yaml_rock)

    def __getitem__(self, key: str) -> Any:
        try:
            return self._yaml_rock[key]
        except KeyError as exc:
            raise MetadataKeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        try:
            self._yaml_rock[key] = value
        except (TypeError, ValueError) as exc:
            raise MetadataValueError(f"Metadata value for `{key}` is invalid.") from exc

    def __delitem__(self, key: str) -> None:
        try:
            del self._yaml_rock[key]
        except KeyError as exc:
            raise MetadataKeyError(key) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._yaml_rock)

    def __len__(self) -> int:
        return len(self._yaml_rock)

    def clear(self) -> None:
        for key in tuple(self._yaml_rock.keys()):
            del self._yaml_rock[key]

    def keys(self) -> Iterable[str]:
        return self._yaml_rock.keys()

    def as_text(self) -> str:
        try:
            return self._yaml_rock.to_yaml().decode()
        except Exception as exc:
            raise MetadataSerializationError("Metadata could not be serialized as YAML.") from exc

    def as_dict(self) -> dict[str, JsonValue]:
        try:
            return dict(self._yaml_rock.to_dict())
        except Exception as exc:
            raise MetadataSerializationError(
                "Metadata could not be serialized as a dictionary."
            ) from exc
