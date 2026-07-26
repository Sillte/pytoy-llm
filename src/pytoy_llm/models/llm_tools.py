import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field, field_validator


class LLMTool(BaseModel, frozen=True):
    impl: Annotated[Callable, Field(description="Implementation of the tool.")]
    doc: str | None = None

    def to_pydantic_tool(self) -> Callable:
        if self.doc is not None:
            try:
                self.impl.__doc__ = self.doc
            except Exception:
                pass
        return self.impl

    @field_validator("impl")
    def _check_callable(cls, value: Callable) -> Callable:
        def _is_allowed_type(tp: Any) -> bool:
            if isinstance(tp, type) and issubclass(tp, BaseModel):
                return True
            if tp in (str, int, float, bool):
                return True

            origin = get_origin(tp)
            if origin in (list, tuple, Sequence, Literal, Union):
                return all(_is_allowed_type(arg) for arg in get_args(tp))
            elif origin in (dict, Mapping):
                k, v = get_args(tp)
                return _is_allowed_type(k) and _is_allowed_type(v)
            elif origin is not None:
                return _is_allowed_type(origin)
            return False

        if not callable(value):
            raise TypeError(f"`{value=}` is not callable.")
        sig = inspect.signature(value)
        for param in sig.parameters.values():
            if param.annotation is inspect._empty:
                raise TypeError("Tool parameters must be type-annotated")
            if not _is_allowed_type(param.annotation):
                raise TypeError(f"Unsupported parameter type: {param.annotation}")

        if sig.return_annotation is inspect._empty:
            raise TypeError("Tool must have return type annotation")
        return value
