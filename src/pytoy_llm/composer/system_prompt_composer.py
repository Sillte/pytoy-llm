import json
import warnings
from collections.abc import Sequence

from pydantic import BaseModel

from pytoy_llm.composer.models import (
    AuxiliaryGuidance,
    OutputSpec,
    SupplementarySectionProtocol,
    SupplementarySections,
    SystemPromptSpec,
)


def _join_blocks(blocks: Sequence[str]) -> str:
    blocks = [block.strip("\n") for block in blocks]
    blocks = [block for block in blocks if block]
    return "\n\n".join(blocks)


class SystemPromptComposer:
    """Compose a system prompt from its semantic specification."""

    def __init__(self, prompt_spec: SystemPromptSpec) -> None:
        self.prompt_spec = prompt_spec

    def compose_prompt(
        self, supplementary_sections: SupplementarySections | None | Sequence[SupplementarySectionProtocol] = None
    ) -> str:
        if supplementary_sections is not None:
            supplementary_sections = SupplementarySections.from_any(supplementary_sections)

        blocks = []

        warn_forbidden_headers(self.prompt_spec.name, min_allowed_header_level=5)

        task_header = f"# Task: {self.prompt_spec.name}"
        task_output_spec = self._to_output_spec_instruction(self.prompt_spec.output_spec)
        blocks = [*blocks, task_header, task_output_spec]

        if self.prompt_spec.intent:
            warn_forbidden_headers(self.prompt_spec.intent, min_allowed_header_level=3)
            blocks = [*blocks, "## Task Intent", self.prompt_spec.intent]

        if self.prompt_spec.rules:
            rules_body = "\n".join(f"* {rule}" for rule in self.prompt_spec.rules)
            blocks = [*blocks, "## Task Rules", rules_body]

        if self.prompt_spec.auxiliary_guidance:
            task_auxiliary_guidance = self._to_auxiliary_guidance_instruction(self.prompt_spec.auxiliary_guidance)
            blocks = [*blocks, task_auxiliary_guidance]

        if supplementary_sections is not None:
            task_supplement = self._to_supplementary_instruction(supplementary_sections=supplementary_sections)
            blocks = [*blocks, task_supplement]

        prompt = _join_blocks(blocks)
        return prompt

    def _to_output_spec_instruction(self, output_spec: OutputSpec) -> str:
        blocks = []
        output_type_explanation = self._to_output_type_explanation(output_spec.output_type)
        warn_forbidden_headers(output_type_explanation, min_allowed_header_level=4)
        blocks = [*blocks, "## Task Output", "### Output Type", output_type_explanation]
        if output_spec.description:
            warn_forbidden_headers(output_spec.description, min_allowed_header_level=4)
            blocks = [*blocks, "### Output Description", output_spec.description]
        return _join_blocks(blocks)

    def _to_auxiliary_guidance_instruction(self, auxiliary_guidance: AuxiliaryGuidance) -> str:
        blocks = [
            "## Auxiliary Guidance",
        ]
        if auxiliary_guidance.guidance_role:
            warn_forbidden_headers(auxiliary_guidance.guidance_role, min_allowed_header_level=4)
            blocks = [*blocks, "### Guidance Role", auxiliary_guidance.guidance_role]

        if auxiliary_guidance.reasoning_guidance:
            warn_forbidden_headers(auxiliary_guidance.reasoning_guidance, min_allowed_header_level=4)
            blocks = [*blocks, "### Reasoning Guidance", auxiliary_guidance.reasoning_guidance]
        return _join_blocks(blocks)

    def _to_supplementary_instruction(self, supplementary_sections: SupplementarySections) -> str:
        blocks = ["## Supplementary Sections"]
        if supplementary_sections.description:
            blocks = [*blocks, supplementary_sections.description]

        section_header_depth = 2 + 1
        for section in supplementary_sections.sections:
            section_explanation = section.compose(header_depth=section_header_depth)
            warn_request_headers(section_explanation, section_header_depth)
            blocks = [*blocks, section_explanation]
        return _join_blocks(blocks)

    def _to_output_type_explanation(self, output_type: type[BaseModel] | type[str]) -> str:
        """Return explanation to include in prompt based on BaseModel schema"""
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            schema_json = json.dumps(output_type.model_json_schema())
            return f"The output must be a JSON matching the following schema:\n```json\n{schema_json}```"
        elif output_type is str:
            return "The output must be as plain text string."
        elif isinstance(output_type, BaseModel):
            return self._to_output_type_explanation(
                output_type.__class__
            )  # Defensive fallback for an unexpected model instance.
        else:
            raise ValueError(f"Invalid output type `{output_type=}`")


def warn_forbidden_headers(text: str, min_allowed_header_level: int | None = 4, skip_first: bool = True) -> None:
    """
    Check each line. Warn if a header is too high (e.g., # or ##)
    compared to the minimum allowed header level.
    """
    if skip_first:
        start = 1
    else:
        start = 0
    for i, line in enumerate(text.splitlines(), start=start):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            if min_allowed_header_level is None:
                warnings.warn(
                    f"Line {i}: `Usage of the header is not recommended here. ",
                    UserWarning,
                )
            else:
                header_level = len(stripped) - len(stripped.lstrip("#"))
                if header_level < min_allowed_header_level:
                    warnings.warn(
                        f"Line {i}: header level {header_level} "
                        f"is below minimum allowed ({min_allowed_header_level}). "
                        "Consider deeper headers for injected structure.",
                        UserWarning,
                    )


def warn_request_headers(text: str, header_depth: int) -> None:
    lines = text.splitlines()
    if not lines:
        warnings.warn("Supplementary section is empty.")
        return

    first_line = lines[0].lstrip()
    expected_prefix = "#" * header_depth + " "

    if not first_line.startswith(expected_prefix):
        warnings.warn(f"Section must start with a Markdown header at depth {header_depth}: `{expected_prefix}<title>`")


if __name__ == "__main__":
    from pytoy_llm.composers.materials import MaterialDataExplorerTaskComposer
    from pytoy_llm.materials.text_files import TextFilesCollector

    collector = TextFilesCollector(
        __file__,
    )
    bundle = collector.bundle
    composer = MaterialDataExplorerTaskComposer([bundle.text_material_data])
    section_text = composer.compose_system_prompt()
    print(section_text)
