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


class SystemPromptComposer:
    """Compose a system prompt from its semantic specification."""

    def __init__(self, prompt_spec: SystemPromptSpec) -> None:
        self.prompt_spec = prompt_spec

    def compose_prompt(
        self, supplementary_sections: SupplementarySections | None | Sequence[SupplementarySectionProtocol] = None
    ) -> str:
        if supplementary_sections is not None:
            supplementary_sections = SupplementarySections.from_any(supplementary_sections)

        warn_forbidden_headers(self.prompt_spec.name, min_allowed_header_level=5)
        task_header = f"# Task: {self.prompt_spec.name}\n\n"

        task_output_spec = self._to_output_spec_instruction(self.prompt_spec.output_spec)

        if self.prompt_spec.intent:
            warn_forbidden_headers(self.prompt_spec.intent, min_allowed_header_level=3)
            task_intent = f"## Task Intent\n\n{self.prompt_spec.intent}\n\n"
        else:
            task_intent = "\n"

        if self.prompt_spec.rules:
            task_rules = "## Task Rules\n\n" + "\n".join(f"* {rule}" for rule in self.prompt_spec.rules) + "\n\n"
        else:
            task_rules = "\n"

        if self.prompt_spec.auxiliary_guidance:
            task_auxiliary_guidance = self._to_auxiliary_guidance_instruction(self.prompt_spec.auxiliary_guidance)
        else:
            task_auxiliary_guidance = ""

        if supplementary_sections is not None:
            task_supplement = self._to_supplementary_instruction(supplementary_sections=supplementary_sections)
        else:
            task_supplement = ""

        # Compose final prompt
        prompt = "\n".join([task_header, task_output_spec, task_intent, task_rules, task_auxiliary_guidance, task_supplement])
        return prompt

    def _to_output_spec_instruction(self, output_spec: OutputSpec) -> str:
        header = "## Task Output\n\n"
        output_type_instruction = self._to_output_type_explanation(output_spec.output_type)

        warn_forbidden_headers(output_type_instruction, min_allowed_header_level=4)
        task_output_type = f"### Output Type\n\n{output_type_instruction}\n\n"
        if output_spec.description:
            warn_forbidden_headers(output_spec.description, min_allowed_header_level=4)
            task_output_description = f"### Output Description\n\n{output_spec.description}\n\n"
        else:
            task_output_description = "\n"

        return "\n".join([header, task_output_type, task_output_description])

    def _to_auxiliary_guidance_instruction(self, auxiliary_guidance: AuxiliaryGuidance) -> str:
        header = "## Auxiliary Guidance\n\n"
        if auxiliary_guidance.guidance_role:
            warn_forbidden_headers(auxiliary_guidance.guidance_role, min_allowed_header_level=4)
            guidance_role = f"### Guidance Role\n\n{auxiliary_guidance.guidance_role}\n\n"
        else:
            guidance_role = ""
        if auxiliary_guidance.reasoning_guidance:
            warn_forbidden_headers(auxiliary_guidance.reasoning_guidance, min_allowed_header_level=4)
            reasoning_guidance = f"### Reasoning Guidance\n\n{auxiliary_guidance.reasoning_guidance}\n\n"
        else:
            reasoning_guidance = ""
        return "\n".join([header, guidance_role, reasoning_guidance])

    def _to_supplementary_instruction(self, supplementary_sections: SupplementarySections) -> str:
        header = "## Supplementary Sections\n\n"
        if supplementary_sections.description:
            description = f"{supplementary_sections.description}\n\n"
        else:
            description = ""
        section_header_depth = 2 + 1
        explanations = []
        for section in supplementary_sections.sections:
            section_explanation = section.compose(header_depth=section_header_depth)
            warn_request_headers(section_explanation, section_header_depth)
            explanations.append(section_explanation)
        explanation_instruction = "\n\n".join(explanations)
        return "\n".join([header, description, explanation_instruction])

    def _to_output_type_explanation(self, output_type: type[BaseModel] | type[str]) -> str:
        """Return explanation to include in prompt based on BaseModel schema"""
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            schema_json = json.dumps(output_type.model_json_schema())
            return f"The output must be a JSON matching the following schema:\n```json\n{schema_json}```\n\n"
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
