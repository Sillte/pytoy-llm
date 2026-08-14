from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, Field, JsonValue

from pytoy_llm.task.models.context import ContextPatch, RuntimeContextPatch
from pytoy_llm.task.models.metas import InvocationSpecMeta


class InvocationInfo(BaseModel, frozen=True):
    kind: Annotated[str, Field(description="Type of invocation")]
    started_at: Annotated[float, Field(description="Start time of this invocation")]
    ended_at: Annotated[float, Field(description="End time of this invocation")]
    meta: Annotated[InvocationSpecMeta, Field(description="Metadata about this invocation spec")] = InvocationSpecMeta()

    @property
    def spec_name(self) -> str:
        return self.meta.name

    @property
    def intent(self) -> str:
        return self.meta.intent


class InvocationTrace(BaseModel, frozen=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: Annotated[Any, Field(description="Input")]
    output: Annotated[Any, Field(description="Output")]
    info: Annotated[InvocationInfo, Field(description="Metatada Information about the invocation.")]
    details: Annotated[Mapping[str, JsonValue], Field(description="detailed information for debugging")] = {}
    children: Annotated[Sequence[InvocationTrace], Field(description="Children of execution")] = ()


@dataclass(frozen=True)
class InvocationResult[T]:
    output: T
    runtime_patch: RuntimeContextPatch | None = None
    context_patch: ContextPatch | None = None
    trace: InvocationTrace | None = None
