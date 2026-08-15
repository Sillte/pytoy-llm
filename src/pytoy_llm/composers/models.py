from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, Field


class OutputSpec[T: BaseModel | str](BaseModel, frozen=True):
    """Specifies the meaning and representation of the LLM's output."""

    output_type: Annotated[
        type[T],
        Field(description="Type or format used to represent the output."),
    ]

    description: Annotated[
        str | None,
        Field(description="Semantic meaning of the expected output."),
    ] = None


class AuxiliaryGuidance(BaseModel, frozen=True):
    """Optional guidance that supports task execution without defining the task itself."""

    reasoning_guidance: Annotated[
        str | None,
        Field(description="Optional guidance for how the task should be reasoned about."),
    ] = None

    guidance_role: Annotated[
        str | None,
        Field(description="Optional role or persona to adopt while performing the task."),
    ] = None


class SystemPromptSpec[T: BaseModel | str](BaseModel):
    """Semantic specification of a task to be expressed as a system prompt."""

    name: Annotated[
        str,
        Field(description="Human-readable name of the task."),
    ]

    output_spec: Annotated[
        OutputSpec[T],
        Field(description="Contract describing the expected output."),
    ]

    intent: Annotated[
        str,
        Field(description="What the LLM is expected to accomplish."),
    ]

    rules: Annotated[
        Sequence[str],
        Field(description="Constraints the LLM must follow while performing the task."),
    ]

    auxiliary_guidance: Annotated[
        AuxiliaryGuidance | None,
        Field(description="Optional guidance that supports task execution."),
    ] = None

    @classmethod
    def from_any(
        cls,
        name: str,
        output_spec: OutputSpec[T] | type[T],
        intent: str | Sequence[str] | None = None,
        rules: str | Sequence[str] | None = None,
        guidance_role: str | None = None,
        reasoning_guidance: str | None = None,
    ) -> Self:
        if not isinstance(output_spec, OutputSpec):
            output_spec = OutputSpec(output_type=output_spec)

        if intent is None:
            intent = ""
        elif isinstance(intent, str):
            pass
        else:
            intent = "\n\n".join(intent)

        if rules is None:
            rules = tuple()
        elif isinstance(rules, str):
            rules = (rules,)

        if guidance_role is not None or reasoning_guidance is not None:
            auxiliary_guidance = AuxiliaryGuidance(
                guidance_role=guidance_role,
                reasoning_guidance=reasoning_guidance,
            )
        else:
            auxiliary_guidance = None
        return cls(name=name, output_spec=output_spec, intent=intent, rules=rules, auxiliary_guidance=auxiliary_guidance)


class SupplementarySectionProtocol(Protocol):
    def compose(self, header_depth: int) -> str:
        """
        Compose this section using the requested header depth.

        The returned text must start with a section title at the
        requested Markdown header depth.

        For example, ``header_depth=3`` requires the result to start
        with ``### <Section Title>``.
        """
        ...


@dataclass(frozen=True)
class SupplementarySections:
    sections: Sequence[SupplementarySectionProtocol]
    description: str | None = None

    @classmethod
    def from_any(cls, arg: Self | Sequence[SupplementarySectionProtocol]) -> Self:
        if isinstance(arg, cls):
            return arg
        return cls(sections=arg)  # type: ignore


type SupplementarySectionsLike = SupplementarySections | Sequence[SupplementarySectionProtocol]
