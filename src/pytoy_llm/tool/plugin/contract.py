from abc import ABC, abstractmethod

from pytoy_llm.models.llm_tools import LLMToolsLike


class LLMToolsPlugin(ABC):
    """Extension contract for a plugin that provides one or more LLM tools."""

    @property
    @abstractmethod
    def tools(self) -> LLMToolsLike:
        """Return the tools provided by this plugin."""
        raise NotImplementedError

    def guidance(self) -> str | None:
        """Return plain-text guidance about when and how to use these tools.

        The caller may include this guidance in a system prompt; it is not
        injected or formatted automatically.
        """
        return None
