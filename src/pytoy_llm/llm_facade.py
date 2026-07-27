from collections.abc import Callable, Sequence

from pydantic import BaseModel

from pytoy_llm.connection_configuration import DEFAULT_NAME
from pytoy_llm.litellm_client.client import PytoyLiteLLMClient
from pytoy_llm.models import LLMMessagesLike
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_messages import LLMResult
from pytoy_llm.models.llm_metas import LLMConfig
from pytoy_llm.models.llm_tools import LLMTool
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


class LLMFacade(BaseModel, frozen=True):
    connection: str | Connection = DEFAULT_NAME
    llm_config: LLMConfig | None = None

    def completion[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
    ) -> T:
        client = PytoyLiteLLMClient(self.connection, llm_config=self.llm_config)
        return client.completion(messages, output_type=output_type)

    def completion_with_result[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
    ) -> LLMResult[T]:
        client = PytoyLiteLLMClient(self.connection, llm_config=self.llm_config)
        return client.completion_with_result(messages, output_type=output_type)

    def run[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
        tools: Sequence[Callable | LLMTool] = (),
    ) -> T:
        """Alias of `run_agent` for better readability."""
        agent = PytoyPydanticAIAgent(self.connection, llm_config=self.llm_config)
        return agent.run(messages, output_type=output_type, tools=tools)

    def run_with_result[T: BaseModel | str](
        self,
        messages: LLMMessagesLike,
        output_type: type[T],
        tools: Sequence[Callable | LLMTool] = (),
    ) -> LLMResult[T]:
        agent = PytoyPydanticAIAgent(self.connection, llm_config=self.llm_config)
        return agent.run_with_result(messages, output_type=output_type, tools=tools)


if __name__ == "__main__":
    facade = LLMFacade()
    result = facade.completion_with_result("Hello", output_type=str)
    result.output.islower

    from pydantic import BaseModel

    class AModel(BaseModel):
        arg: int

        def is_valid(self) -> bool:
            return True

    facade = LLMFacade()
    result = facade.completion_with_result("Hello", output_type=AModel)
