from pathlib import Path
from typing import Any, Mapping, Self, Sequence

from pydantic import BaseModel

from pytoy_llm.models import LLMTokens
from pytoy_llm.models.llm_messages import LLMMessage
from pytoy_llm.models.parts import PartAdapter


class LLMMessageView(BaseModel, frozen=True):
    message: LLMMessage
    llm_tokens: LLMTokens | None


class LLMMessageMaterial(BaseModel, frozen=True):
    """Represents the set of LLMMessages."""

    part_scheme: Mapping[str, Any] = PartAdapter.json_schema()
    message_scheme: Mapping[str, Any] = LLMMessage.model_json_schema()

    views: Sequence[LLMMessageView]

    @classmethod
    def from_any(
        cls,
        llm_messages: LLMMessage | Sequence[LLMMessage],
        llm_tokens: LLMTokens | None | Sequence[LLMTokens | None],
    ) -> Self:
        if isinstance(llm_messages, LLMMessage):
            llm_messages = [llm_messages]
        if llm_tokens is None or isinstance(llm_tokens, LLMTokens):
            llm_tokens = [llm_tokens]
        views = [
            LLMMessageView(message=message, llm_tokens=llm_tokens)
            for message, llm_tokens in zip(llm_messages, llm_tokens, strict=True)
        ]
        return cls(views=views)

    def dump(self, file_path: Path | str) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.model_dump_json(indent=2))
