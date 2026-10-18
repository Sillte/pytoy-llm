from collections.abc import Sequence

from pytoy_llm.models.llm_messages import LLMMessage


class LLMCancelException(Exception):
    """Raised after an LLM call completes when cancellation was requested."""

    def __init__(self, messages: Sequence[LLMMessage]) -> None:
        self.messages = tuple(messages)
        super().__init__("LLM execution was canceled.")
