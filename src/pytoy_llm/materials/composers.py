from collections.abc import Sequence

from pytoy_llm.composers import InvocationComposer, SystemPromptSpec
from pytoy_llm.materials.models import MaterialData, MaterialSection, MaterialUsage
from pytoy_llm.task.models import LLMInvocationSpec, TaskSpec


class MaterialDataExplorerTaskComposer:
    """Compose and optionally analyze materials with an LLM."""

    def __init__(self, materials: Sequence[MaterialData], *, system_prompt_spec: SystemPromptSpec | None = None) -> None:
        self._materials = materials
        self._system_prompt_spec = system_prompt_spec or self._get_system_prompt_spec()

        self._invocation_composer = InvocationComposer(self._system_prompt_spec)

        usage = MaterialUsage(
            usage=(
                "Use this material as evidence and cross-reference it with other "
                "materials where relevant. Do not treat unsupported interpretations "
                "as facts."
            )
        )
        self._sections = [
            MaterialSection(name=f"DataType {i}", usage=usage, data=m_data) for i, m_data in enumerate(self._materials)
        ]
        self._supplementary_sections = MaterialSection.build_supplementary_sections(self._sections)

    def compose_system_prompt(self) -> str:
        return self._invocation_composer.system_prompt_composer.compose_prompt(
            supplementary_sections=self._supplementary_sections
        )

    def compose_llm_invocation_spec(self) -> LLMInvocationSpec:
        return self._invocation_composer.compose_llm_invocation_spec(supplementary_sections=self._supplementary_sections)

    def compose_task_spec(self) -> TaskSpec:
        return TaskSpec.from_single_spec(self.compose_llm_invocation_spec())

    def _get_system_prompt_spec(self) -> SystemPromptSpec:
        return SystemPromptSpec.from_any(
            name="Explore Materials",
            intent=(
                "Examine the provided materials and identify useful findings, "
                "patterns, relationships, anomalies, and implications supported "
                "by the materials."
            ),
            rules=[
                "Base findings on the provided materials.",
                "Distinguish observations from interpretations.",
                "Do not invent facts that are not supported by the materials.",
                "Prefer specific findings over generic summaries.",
                "When evidence is insufficient, state the uncertainty explicitly.",
            ],
            output_spec=str,
        )
