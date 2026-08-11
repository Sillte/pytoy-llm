from pydantic import BaseModel, Field
import textwrap
import itertools
from typing import runtime_checkable, Self


class TextSemanticsOutcomeModel(BaseModel, frozen=True):
    """Semantics OUTCOME of the given text.

    `description` is a natural-language constraint layer.
    Structured parameters may be added later when stable,
    operationally useful distinctions have been identified.

    When the instruction in `description` contradicts a structured
    parameter, the structured parameter takes precedence.
    """

    description: str = Field(
        description="Description of the semantic outcome of the text."
    )


class TextPragmaticsOutcomeModel(BaseModel, frozen=True):
    """Pragmatics OUTCOME of the given text.

    `description` is a natural-language constraint layer.
    Structured parameters may be added later when stable,
    operationally useful distinctions have been identified.

    When the instruction in `description` contradicts a structured
    parameter, the structured parameter takes precedence.
    """

    description: str = Field(
        description="Description of the pragmatic outcome of the text."
    )


class TextOutcomeModel(BaseModel, frozen=True):
    """`OUTCOME` property of the given text.

    OUTCOME is defined as:

    A representation of what the text communicates, asserts,
    presupposes, commits to, or otherwise establishes as a
    linguistically relevant consequence of the utterance.

    For analysis, OUTCOME is described using SEMANTICS and PRAGMATICS.
    """

    semantics: TextSemanticsOutcomeModel = Field(
        description="Semantic outcome of the text."
    )

    pragmatics: TextPragmaticsOutcomeModel = Field(
        description="Pragmatic outcome or communicative properties of the text."
    )


class TextRealizationModel(BaseModel, frozen=True):
    """`REALIZATION` property of the given text.

    REALIZATION is defined as:

    The linguistic and discourse-level choices and mechanisms
    through which an OUTCOME is expressed, conveyed, or interpreted.

    This class describes the means or strategies used to realize
    a `TextOutcomeModel`.

    `description` is a natural-language constraint layer.
    Structured parameters may be added later when stable,
    operationally useful distinctions have been identified.

    When the instruction in `description` contradicts a structured
    parameter, the structured parameter takes precedence.
    """

    description: str = Field(
        description=(
            "Natural-language description of the linguistic "
            "and discourse-level means used to realize the outcome."
        )
    )


