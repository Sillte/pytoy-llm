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

    The transformed text must not add, remove, or alter any proposition,
    event, participant, temporal relation, spatial relation, or other
    semantic relation directly supported by the original text.

    In particular, preserve absolutely the core truth-conditional content
    or minimal truth-conditional content identifiable and directly supported in the
    original text, insofar as such content can be determined.

    If satisfying the transformation instruction requires a change
    to the semantic content, prioritize preservation of the original
    semantic content and make the smallest necessary change.
    If necessary change requested by instruction contradicts
    core truth-conditional content or minimal truth-conditional content directly supported in the text, 
    prioritize the preservation of them, disregard the conflicting part of the instruction.

    Note that linguistic terms such as `core truth-conditional content`
    and `minimal truth-conditional content` are used as conceptual guidance
    rather than as a requirement for a formally exact semantic analysis.
    """.strip()


class WeakSemanticPreservationRule(BaseTransformRule, frozen=True):
    """Request to maintain the semantic content of the text to some extent."""

    rule: str = """
    Preserve the original semantic content where it does not conflict
    with the transformation instruction.

    In particular, preserve the core truth-conditional content
    or minimal truth-conditional content identifiable in the
    original text, insofar as such content can be determined.

    When the transformation instruction conflicts with the original
    semantic content, prioritize satisfying the instruction and allow
    the necessary changes to the semantic content, including changes
    to the core truth-conditional content or minimal truth-conditional
    content when required.

    Prioritize the semantics properties directly supported in the text over the semantic properties inferred.

    Note that linguistic terms such as `core truth-conditional content`
    and `minimal truth-conditional content` are used as conceptual guidance
    rather than as a requirement for a formally exact semantic analysis.
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

    If the transformation instruction conflicts with an original
    pragmatic property, prioritize preservation of that pragmatic property
    unless the instruction explicitly requires changing it.

    Prioritize preservation of pragmatic properties that are directly
    supported by the original text over properties that depend primarily
    on contextual inference.

    Note that pragmatic properties may depend on contextual interpretation.
    They should be preserved insofar as they can reasonably be determined
    from the original text and the provided context.
    """.strip()


class WeakPragmaticPreservationRule(BaseTransformRule, frozen=True):
    """Request to maintain the pragmatic properties of the text weakly."""

    rule: str = """
    Preserve the original pragmatic OUTCOME as much as possible.

    The transformed text should preserve the original communicative function,
    including the speech act, intended direction of action, interpersonal stance,
    degree of directness, urgency, politeness, and other pragmatic properties
    represented in the original OUTCOME unless specified by the instruction. 

    If satisfying the transformation instruction requires a change
    to the pragmatic outcome, make the necessary changes.

    If the transformation instruction conflicts with an original
    pragmatic property, prioritize the instruction over preservation of the original pragmatic OUTCOME.  
    Prioritize preservation of the pragmatic properties explicitly represented in the original pragmatic OUTCOME
    over preservation of the inferred pragmatic properties in the original pragmatic OUTCOME. 

    Note that pragmatic properties may depend on contextual interpretation.
    They should be preserved insofar as they can reasonably be determined
    from the original text and the provided context.
    """


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
