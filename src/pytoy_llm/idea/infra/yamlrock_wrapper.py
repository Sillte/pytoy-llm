from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Self, cast

import yamlrocks
from yamlrocks import YAMLRocksDocument


class YamlRockWrapper:
    def __init__(self, yaml_rock: YAMLRocksDocument | None = None) -> None:
        yaml_rock = yaml_rock or cast(
            YAMLRocksDocument, yamlrocks.loads("{}", option=yamlrocks.OPT_ROUND_TRIP)
        )
        self._yaml_rock = yaml_rock

    @classmethod
    def from_yaml_text(cls, yaml_text: str) -> Self:
        yaml_rock = cast(
            YAMLRocksDocument,
            yamlrocks.loads(
                yaml_text,
                option=yamlrocks.OPT_ROUND_TRIP,
            ),
        )
        return cls(yaml_rock)

    def __getitem__(self, key: str) -> Any:
        return self._yaml_rock[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._yaml_rock[key] = value

    def __delitem__(self, key: str) -> None:
        del self._yaml_rock[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._yaml_rock)

    def __len__(self) -> int:
        return len(self._yaml_rock)

    def clear(self) -> None:
        for key in tuple(self._yaml_rock.keys()):
            del self._yaml_rock[key]

    def keys(self) -> Iterable[str]:
        return self._yaml_rock.keys()

    def to_text(self) -> str:
        return self._yaml_rock.to_yaml().decode()
