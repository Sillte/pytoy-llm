from typing import Sequence

from typing import Callable, Mapping, assert_never, overload, Literal
from pydantic_ai import Agent
from pydantic import BaseModel

from pydantic_ai.models import Model as PydanticAIModel
from pytoy_llm.models import (
    Connection,
    InputMessage,
    LLMOutputModel,
    ResultType,
    LLMTool,
    LLMConfig,
)
from pytoy_llm.connection_configuration import ConnectionConfiguration, DEFAULT_NAME
from pydantic_ai import UserPromptPart, SystemPromptPart
from pydantic_ai import ModelResponse, TextPart, AgentRunResult, ModelRequest

class PydanticAIModelFactory:
    @staticmethod
    def create(connection: Connection, llm_config: LLMConfig) -> PydanticAIModel:
        base_url = connection.base_url
        model_name = connection.model
        api_key = connection.api_key
        model_settings = llm_config.to_pydantic_model_settings()

        parts = model_name.split("/")
        if len(parts) < 1:
            raise ValueError(f"Invalid model name {model_name}")

        if parts[0] == "gemini":
            assert base_url.find("google") != -1, "for fool proof."
            from pydantic_ai.providers.google import GoogleProvider
            from pydantic_ai.models.google import GoogleModel
            # For Google, `base_url` must not be passed.
            provider = GoogleProvider(api_key=api_key)
            sub_name = "/".join(parts[1:])
            return GoogleModel(sub_name, provider=provider, settings=model_settings)
        elif parts[0] in {"openai"}:
            assert base_url, "for fool proof."
            from pydantic_ai.providers.openai import OpenAIProvider
            from pydantic_ai.models.openai import OpenAIChatModel
            # For Google, `openai` or in local LLM, you must pass the url.
            provider = OpenAIProvider(api_key=api_key, base_url=base_url)
            sub_name = "/".join(parts[1:])
            return OpenAIChatModel(sub_name, provider=provider, settings=model_settings)
        else:
            assert base_url, "for fool proof."
            from pydantic_ai_litellm import LiteLLMModel
            return LiteLLMModel(
                model_name=model_name, api_key=api_key, api_base=base_url, settings=model_settings
            )



class PytoyAgent:
    def __init__(
        self,
        connection: str | Connection,
        llm_config: LLMConfig | None = None,
    ) -> None:
        llm_config = llm_config or LLMConfig()

        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)
        self._connection = connection
        self._llm_config = llm_config

    def _normalize_tool(self, tool: LLMTool | Callable) -> Callable:
        if isinstance(tool, LLMTool):
            return tool.to_pydantic_tool()
        else:
            return tool

    def _make_agent(self, system_prompts: Sequence[str], tools: Sequence[LLMTool | Callable]):
        model = PydanticAIModelFactory.create(self._connection, self._llm_config)
        tools = [self._normalize_tool(tool) for tool in tools]
        return Agent(model=model, system_prompt=system_prompts, tools=tools)

    @overload
    def run_sync[T: BaseModel | str](
        self,
        content: str | InputMessage | Sequence[InputMessage | str | Mapping],
        output_type: type[T],
        tools: Sequence[LLMTool | Callable] = tuple(),
        result_type: Literal["output"] = "output",
    ) -> T:
        ...

    @overload
    def run_sync[T: BaseModel | str](
        self,
        content: str | InputMessage | Sequence[InputMessage | str | Mapping],
        output_type: type[T],
        tools: Sequence[LLMTool | Callable] = tuple(),
        result_type: Literal["native-result"] = "native-result",
    ) -> AgentRunResult[T]:
        ...

    @overload
    def run_sync[T: BaseModel | str](
        self,
        content: str | InputMessage | Sequence[InputMessage | str | Mapping],
        output_type: type[T],
        tools: Sequence[LLMTool | Callable] = tuple(),
        result_type: Literal["pytoy-result"] = "pytoy-result",
    ) -> LLMOutputModel[T]:
        ...

    def run_sync[T: BaseModel | str](
        self,
        content: str | InputMessage | Sequence[InputMessage | str | Mapping],
        output_type: type[T] = str,
        tools: Sequence[LLMTool | Callable] = tuple(),
        result_type: ResultType = "output",
    ) ->  T | AgentRunResult[T] | LLMOutputModel[T]:
        input_messages = InputMessage.to_messages(content)


        # Remove system_prompts.
        system_prompts = [item.content for item in input_messages if item.role == "system"]
        input_messages = [item for item in input_messages if item.role != "system"]

        last_user_index = None
        for index in reversed(range(len(input_messages))):
            mes = input_messages[index]
            if mes.role == "user":
                last_user_index = index
                break
        if last_user_index is None:
            user_prompt = None
        else:
            user_prompt = input_messages[last_user_index].content
            input_messages = (
                input_messages[:last_user_index] + input_messages[last_user_index + 1 :]
            )

        def _convert(message: InputMessage):
            if message.role == "user":
                return ModelRequest(parts=[UserPromptPart(content=message.content)])
            elif message.role == "system":
                return ModelRequest(parts=[SystemPromptPart(content=message.content)])
            elif message.role == "assistant":
                return ModelResponse(parts=[TextPart(content=message.content)])
            else:
                raise ValueError(f"`{message=}` is invalid.")

        history = [_convert(item) for item in input_messages]
        agent = self._make_agent(system_prompts=system_prompts, tools=tools)

        result = agent.run_sync(
            user_prompt=user_prompt, output_type=output_type, message_history=history
        )

        match result_type:
            case "native-result":
                return result
            case "pytoy-result":
                return LLMOutputModel.from_pydantic_run_result(result, input_messages)
            case "output":
                return result.output
            case _:
                assert_never(result_type)



def experiment_func(name: str = DEFAULT_NAME):
    class AnswerOutput(BaseModel):
        summary: str
        key_points: list[str]

    from pytoy_llm.models import InputMessage

    mes = InputMessage(role="user", content="Are you happy?")
    config = LLMConfig(temperature=0.7)
    agent = PytoyAgent(name, llm_config=config)
    ret = agent.run_sync(
        content=[mes], output_type=AnswerOutput, result_type="output"
    )
    print(ret)


if __name__ == "__main__":
    experiment_func()
