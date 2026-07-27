from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel, Field


class LLMTaskSpecMeta(BaseModel):
    name: Annotated[str, Field(description="Human-readable task name")]
    intent: Annotated[str | None, Field(description="What the overall task is intended to do")] = None
    rules: Annotated[Sequence[str] | None, Field(description="Guiding rules or constraints for this task")] = None
    description: Annotated[str | None, Field(description="Optional longer explanation of the task purpose")] = None


class InvocationSpecMeta(BaseModel, frozen=True):
    name: Annotated[str, Field(description="Name of the invocation step.")] = "No name"
    intent: Annotated[str, Field(description="Intent of this invocation step.")] = "No description"
