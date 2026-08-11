from pydantic import BaseModel, Field

from pytoy_llm.event_sinks import EventSinkProtocol
from pytoy_llm.materials.composers.invocation_prompt_composer import InvocationPromptComposer, SystemPromptTemplate
from pytoy_llm.task import TaskExecutor, TaskRequest
from pytoy_llm.task.models import TaskSpec
from pytoy_llm.texts.models import TextOutcomeModel, TextRealizationModel


class TextAnalysisModel(BaseModel, frozen=True):
    """Analysis of a text in terms of OUTCOME and REALIZATION.

    OUTCOME is not a summary of the text.

    OUTCOME is an intermediate representation used for
    subsequent text transformation.

    Preserve information that may be necessary to reconstruct
    the original communicative content, including but not limited to:
    - propositions and events
    - participants and their relations
    - temporal and spatial information
    - modality
    - negation
    - repetition or recurrence
    - presuppositions and implications
    - speech acts such as questions, requests, commands, warnings
    - the speaker's communicative commitments

    Do not replace concrete information with a more general
    description merely because the general description captures
    the overall purpose of the utterance.
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
