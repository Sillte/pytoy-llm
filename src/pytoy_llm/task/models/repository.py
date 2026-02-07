from pydantic import BaseModel, PrivateAttr


from typing import Any, Mapping


class LLMTaskStateRepository(BaseModel):
    _data: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def snapshot(self) -> dict[str, Any]:
        return self._data.copy()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, updates: Mapping[str, Any]) -> None:
        self._data.update(updates)