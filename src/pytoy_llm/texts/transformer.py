from typing import Self

from pydantic import BaseModel, Field

from pytoy_llm.event_sinks import EventSinkProtocol
from pytoy_llm.materials.composers.invocation_prompt_composer import InvocationPromptComposer, SystemPromptTemplate
from pytoy_llm.task import TaskExecutor, TaskRequest
from pytoy_llm.task.models import TaskSpec
from pytoy_llm.texts.analyzer import TextAnalysisModel, TextAnalyzer
from pytoy_llm.texts.models import TextOutcomeModel, TextRealizationModel


class BaseTransformRule(BaseModel, frozen=True):
    rule: str = Field(description="The rule of transformation of the text.")


class StrongSemanticPreservationRule(BaseTransformRule, frozen=True):
    """Request to maintain the semantic content of the text strongly."""

    rule: str = """
    Preserve the original semantic content as completely as possible.

    The transformation must preserve the semantic content represented
    in the original semantic OUTCOME, including propositions, events,
    participants and their relations, temporal and spatial relations,
    modality, negation, repetition or recurrence, and other semantic
    relations relevant to the original text.

    Semantic content represented as `explicit` should be preserved
    without alteration whenever possible. Semantic content represented
    as `inferred` should also be preserved insofar as it constitutes
    a semantic interpretation supported by the text.

    If satisfying the transformation instruction requires a change
    to the semantic content, prioritize preservation of the original
    semantic content and make the smallest necessary change.

    If the requested transformation conflicts with `explicit` semantic content
    that must be preserved, prioritize preservation of that content
    and disregard the conflicting part of the instruction.

    If the requested transformation conflicts with `inferred` semantic content
    that should be preserved, prioritize the requested transformation
    to the extent that it remains consistent with the `explicit` semantic
    content, and make the smallest necessary change.

    The distinction between `explicit` and `inferred` describes
    how semantic content is recovered in the analysis; it does not
    itself determine the strength of semantic preservation.
    """.strip()


class WeakSemanticPreservationRule(BaseTransformRule, frozen=True):
    """Request to maintain the semantic content of the text to some extent."""

    rule: str = """
    Preserve the original semantic content to the extent reasonably possible.

    The transformation should preserve the semantic content represented
    in the original semantic OUTCOME, including propositions, events,
    participants and their relations, temporal and spatial relations,
    modality, negation, repetition or recurrence, and other semantic
    relations relevant to the original text.

    In particular, preserve the core truth-conditional content or
    minimal semantic content directly supported by the text, insofar
    as such content can be determined.

    If satisfying the transformation instruction requires a change
    to the semantic content, prioritize the transformation instruction
    over semantic content that is not essential to the core
    truth-conditional or minimal semantic content, and make the
    smallest necessary change.

    If the requested transformation conflicts with `explicit` semantic
    content that constitutes core truth-conditional or minimal semantic
    content, prioritize preservation of that content and disregard
    the conflicting part of the instruction.

    If the requested transformation conflicts with `explicit` semantic
    content that is regarded not to constitute core truth-conditional or
    minimal semantic content, prioritize the requested transformation,
    making the smallest necessary change to the `explicit` semantic content
    while maintaining the core truth-conditional or minimal semantic content.

    If the requested transformation conflicts only with `inferred`
    semantic content, prioritize the requested transformation while
    preserving that inferred content where it remains compatible
    with the transformation.

    The distinction between `explicit` and `inferred` describes
    how semantic content is recovered in the analysis; it does not
    itself determine the strength of semantic preservation.

    Note that linguistic terms such as ``core truth-conditional content``
    and ``minimal semantic content`` are used as conceptual guidance
    rather than as requirements for a formally exact semantic analysis.
""".strip()


class StrongPragmaticPreservationRule(BaseTransformRule, frozen=True):
    """Request to maintain the pragmatic properties of the text strongly."""

    rule: str = """
    Preserve the original pragmatic OUTCOME as completely as possible.

    The transformed text should preserve the original communicative
    function and pragmatic properties represented in the original OUTCOME,
    including, where applicable, the speech act, intended direction of action,
    interpersonal stance, degree of directness, urgency, politeness,
    and other pragmatic properties.

    Do not introduce a substantially different communicative act,
    interpersonal stance, or communicative commitment unless required
    by the transformation instruction.

    If satisfying the transformation instruction requires a change
    to the pragmatic outcome, make the smallest necessary change.

    If the transformation instruction conflicts with an `explicit`
    pragmatic property, prioritize preservation of that content
    and disregard the conflicting part of the instruction.

    If the transformation instruction conflicts with an `inferred`
    pragmatic property, prioritize the requested transformation while
    making the smallest necessary change to that `inferred` pragmatic
    property, insofar as the result remains consistent with the
    `explicit` pragmatic properties.

    The distinction between `explicit` and `inferred` describes
    how pragmatic properties are recovered in the analysis; it does not
    itself determine the strength of pragmatic preservation.
    """.strip()


class WeakPragmaticPreservationRule(BaseTransformRule, frozen=True):
    """Request to maintain the pragmatic properties of the text weakly."""

    rule: str = """
    Preserve the original pragmatic content to the extent reasonably possible.

    The transformed text should preserve the original communicative
    function and pragmatic properties represented in the original OUTCOME,
    including, where applicable, the speech act, intended direction of action,
    interpersonal stance, degree of directness, urgency, politeness,
    and other pragmatic properties.

    If satisfying the transformation instruction requires a change
    to the pragmatic outcome, make the smallest necessary changes.

    If the transformation instruction conflicts with an `explicit`
    pragmatic property, prioritize the requested transformation.
    However, preserve that pragmatic property to the extent reasonably
    possible, for example by making the smallest necessary changes
    while retaining enough of the original property for it to remain
    recoverable from the transformed text.

    If the transformation instruction conflicts with an `inferred`
    pragmatic property, prioritize the requested transformation while
    preserving that property where it remains compatible with the
    transformation and the `explicit` pragmatic properties.

    The distinction between `explicit` and `inferred` describes
    how pragmatic properties are recovered in the analysis; it does not
    itself determine the strength of pragmatic preservation.
""".strip()


class CompositeTransformationRule(BaseTransformRule, frozen=True):
    """Composite rules for text transformation."""

    rule: str

    @classmethod
    def from_rules(cls, semantics_rule: BaseTransformRule | None, pragmatics_rule: BaseTransformRule | None) -> Self:
        rules = []
        if semantics_rule:
            rules.append(
                "\n\n".join(
                    ["For transforming semantics OUTCOME, please comply with the following rules:", semantics_rule.rule]
                )
            )
        if pragmatics_rule:
            rules.append(
                "\n\n".join(
                    ["For transforming pragmatics OUTCOME, please comply with the following rules:", pragmatics_rule.rule]
                )
            )
        if rules:
            rule = "\n\n".join(rules)
        else:
            rule = "No rules for transformation."
        return cls(rule=rule)


class TextTransformRequest(BaseModel, frozen=True):
    """Request for transforming the text"""

    orig_text: str = Field(description="Original text")
    orig_outcome: TextOutcomeModel
    orig_realization: TextRealizationModel | None = Field(description="Original realization if given.", default=None)
    transform_rule: str = Field(description="Rule for transformation")
    instruction: str = Field(
        description="Instruction for transforming the text. As long as transform_rule permits, this instruction determines what modification should be performed on the `orig_text`."
    )


class TextTransformer:
    def __init__(self, event_sink: None | EventSinkProtocol = None) -> None:
        self._event_sink = event_sink

    def transform(self, text: str, transform_rule: BaseTransformRule | str, instruction: str) -> str:
        analyzer = TextAnalyzer()
        analysis = analyzer.analyze(text)
        return self.transform_from_analysis(analysis, transform_rule=transform_rule, instruction=instruction)

    def transform_from_analysis(
        self, analysis: TextAnalysisModel, transform_rule: BaseTransformRule | str, instruction: str
    ) -> str:

        if isinstance(transform_rule, BaseTransformRule):
            transform_rule = transform_rule.rule

        template = SystemPromptTemplate(
            name="TextTransform",
            intent="Transform the text based on instruction.",
            rules=["Return only the transformed text.", "Do not include other information such as reasoning."],
            output_description="Transform the text",
            output_type=str,
            reasoning_guidance="Please refer to JSON Schema",
            role="Expert of the linguistics and education teacher.",
        )

        composer = InvocationPromptComposer(prompt_template=template)
        llm_spec = composer.compose_llm_invocation_spec()

        transform_request = TextTransformRequest(
            orig_text=analysis.text,
            orig_outcome=analysis.outcome,
            orig_realization=analysis.realization,
            instruction=instruction,
            transform_rule=transform_rule,
        )

        task_spec = TaskSpec.from_single_spec("TextTransform", llm_spec)
        request = TaskRequest(spec=task_spec, input=transform_request.model_dump_json(indent=2))

        response = TaskExecutor().execute(request, event_sink=self._event_sink)
        return response.output


if __name__ == "__main__":
    pass
