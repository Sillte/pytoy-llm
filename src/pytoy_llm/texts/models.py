from pydantic import BaseModel, Field


class TextSemanticsOutcomeModel(BaseModel, frozen=True):
    """Semantics OUTCOME of the given text.

    The model separates semantic content that is directly supported
    by the text from semantic content obtained through interpretation.

    ``explicit`` roughly corresponds to content directly encoded or
    conventionally expressed by the linguistic form, while ``inferred``
    corresponds to content recoverable through semantic interpretation
    but not directly stated in the text.

    This distinction is intentionally operational rather than a claim
    about a strict boundary in linguistic theory.

    In particular, ``explicit`` should be understood as broadly
    corresponding to explicitly represented or asserted semantic content,
    while ``inferred`` may include semantically licensed inferences
    or implications.

    As a guideline, ``explicit`` should tend toward minimal propositions
    directly supported by the text, while ``inferred`` should contain
    additional content that follows through interpretation or inference,
    which may tend toward the boundary against ``pragmatics`.
    """

    explicit: str = Field(description=("Description of semantic content directly supported by the text."))
    inferred: str = Field(description=("Description of semantic content inferred from the text but not directly expressed."))


class TextPragmaticsOutcomeModel(BaseModel, frozen=True):
    """Pragmatics OUTCOME of the given text.

    The model separates pragmatic properties that are directly
    supported by the text from pragmatic interpretations that
    require additional contextual or intentional inference.

    ``explicit`` roughly corresponds to pragmatic properties that
    are directly recoverable from the linguistic form and its
    conventional use, such as communicative functions or
    interpersonal properties.

    ``inferred`` corresponds to pragmatic interpretations that
    depend more strongly on contextual assumptions, speaker
    intentions, or other information beyond the text itself.

    This distinction is intentionally operational rather than a
    claim about a strict boundary in linguistic theory.

    As a guideline, ``explicit`` should tend toward pragmatic
    properties that are directly recoverable from the text,
    while ``inferred`` should tend toward interpretations that
    depend on context, intention, or information outside the text.
    """

    explicit: str = Field(description=("Description of pragmatic properties directly recoverable from the text."))
    inferred: str = Field(
        description=(
            "Description of pragmatic interpretations that require contextual, intentional, or other inferential information."
        )
    )


class TextOutcomeModel(BaseModel, frozen=True):
    """OUTCOME property of the text.

    OUTCOME is a representation of what the text communicates,
    asserts, presupposes, commits to, or otherwise establishes as a
    linguistically relevant consequence of the text.

    OUTCOME is described from two complementary perspectives: SEMANTICS and PRAGMATICS.

    SEMANTICS describes semantic content conveyed by the text,
    while PRAGMATICS describes communicative functions,
    interpersonal properties, and other pragmatic consequences
    of using the text.

    Each perspective further separates ``explicit`` content from
    ``inferred`` content. These labels are operational distinctions
    describing how directly content can be recovered within each
    perspective, rather than linguistic categories.

    ``semantics.inferred`` may contain semantic interpretations
    that are not directly expressed but follow through semantic
    interpretation. In contrast, ``pragmatics.explicit`` may contain
    pragmatic properties directly recoverable from the text's
    linguistic form and conventional use, even when those properties
    are not themselves semantic content expressed by the text.

    Therefore, ``semantics.inferred`` and ``pragmatics.explicit``
    are not equivalent categories.

    These distinctions are operational rather than strict linguistic
    categories. They should be used to preserve information relevant
    to the text for the purposes such as interpretation, analysis, transformation.
    These distinctions allow uncertainty at the boundary between linguistic form,
    semantic interpretation, and pragmatic interpretation.

    ``semantics.inferred`` and ``pragmatics.explicit``should therefore not be interpreted
    as requiring a strict or mutually exclusive partition of all information in the text.
    """

    semantics: TextSemanticsOutcomeModel = Field(description="Semantic outcome of the text.")

    pragmatics: TextPragmaticsOutcomeModel = Field(description="Pragmatic outcome of the text.")

    text: str = Field(description="Text from which the outcome is derived.")


class TextRealizationModel(BaseModel, frozen=True):
    """REALIZATION property of the given text.

    ```
    REALIZATION describes the linguistic and discourse-level
    choices and mechanisms through which an OUTCOME is expressed,
    conveyed, or interpreted.

    It describes how the text realizes its OUTCOME rather than
    what the OUTCOME itself is.

    The description may include features such as:
    - lexical choices
    - grammatical constructions
    - sentence structure
    - modality and politeness marking
    - register and style
    - discourse organization
    - repetition, emphasis, or omission
    - figurative or rhetorical devices
    - etc.
    """

    description: str = Field(
        description=("Natural-language description of the linguistic and discourse-level means used to realize the outcome.")
    )
