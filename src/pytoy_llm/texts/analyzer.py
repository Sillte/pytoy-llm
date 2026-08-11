from pydantic import BaseModel, Field

from pytoy_llm.event_sinks import EventSinkProtocol
from pytoy_llm.materials.composers.invocation_prompt_composer import InvocationPromptComposer, SystemPromptTemplate
from pytoy_llm.task import TaskExecutor, TaskRequest
from pytoy_llm.task.models import TaskSpec
from pytoy_llm.texts.models import TextOutcomeModel, TextRealizationModel


class TextAnalysisModel(BaseModel, frozen=True):
    """Analysis of a text in terms of OUTCOME and REALIZATION.

    OUTCOME is NOT a summary of the text,
    but an intermediate representation used for reconstructing
    the original content and communicative properties of the whole text.
    REALIZATION are linguistic methods to convey the semantics and pragmatics
    of OUTCOME.

    Represent the text at the level of specific propositions, events,
    participants, relations, and communicative acts expressed or
    directly supported by the text.

    Preserve information that may be necessary to reconstruct
    the original communicative content, including but not limited to:
    - propositions and events
    - participants and their semantic or discourse relations
    - temporal and spatial relations
    - modality and degrees of certainty or obligation
    - negation
    - repetition or recurrence
    - presuppositions and implications
    - speech acts such as questions, requests, commands, and warnings
    - the direction of communicative acts
    - the speaker's communicative commitments

    Do not replace concrete information with a more general
    description merely because the general description captures
    the overall purpose of the text.

    Do not infer a more general proposition, event, participant,
    or communicative act when a more specific interpretation is
    directly supported by the text.

    Preserve distinctions between:
    - what is explicitly expressed,
    - what is directly recoverable,
    - what is directly implied or presupposed,
    - and what is merely plausible from context.

    Do not treat a contextual inference as explicitly expressed
    content unless the text itself supports that interpretation.

    For communicative acts, represent not only the type of act
    but also its relevant participants and direction, such as
    who is requesting, commanding, warning, or informing whom.

    For semantic content, preserve the relations among participants,
    events, actions, conditions, and outcomes rather than reducing
    them to a general statement of intent.

    OUTCOME should contain the most specific content that can be
    reasonably recovered from the text without requiring speculative
    interpretation.
    """

    outcome: TextOutcomeModel
    realization: TextRealizationModel
    text: str = Field(description="Original text.")


class TextAnalyzer:
    def __init__(self, event_sink: None | EventSinkProtocol = None) -> None:
        self._event_sink = event_sink
        pass

    def analyze(self, text: str) -> TextAnalysisModel:
        template = SystemPromptTemplate(
            name="TextAnalyzer",
            intent="Analyze the text and generate `TextAnalysisModel`",
            rules=(),
            output_description="Analyzed model",
            output_type=TextAnalysisModel,
            reasoning_guidance="Please refer to JSON Schema",
            role="Expert of the linguistics and education teacher.",
        )
        composer = InvocationPromptComposer(prompt_template=template)
        llm_spec = composer.compose_llm_invocation_spec()
        request = TaskRequest(spec=TaskSpec.from_single_spec("TextAnalyzer", llm_spec), input=text)
        response = TaskExecutor().execute(request, event_sink=self._event_sink)
        return response.output


if __name__ == "__main__":
    pass
