from typing import Sequence

from typing import Callable, Mapping
from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models import Model as PydanticAIModel
from pydantic_ai import RunContext
from pytoy_llm.models import Connection, InputMessage, LLMOutputModel, SyncOutputType, SyncResultClass, LLMTool, LLMConfig
from pytoy_llm.connection_configuration import ConnectionConfiguration, DEFAULT_NAME
from pydantic_ai import UserPromptPart,  SystemPromptPart
from pydantic_ai import ModelResponse, TextPart, AgentRunResult, ModelSettings


def get_model(model_name: str, api_key: str, base_url: str, model_settings: ModelSettings) -> PydanticAIModel:
    parts = model_name.split("/")
    if len(parts) < 1:
        raise ValueError(f"Invalid model name {model_name}")

    if parts[0] == "gemini":
        # For Google, `base_url` must not be passed.
        provider = GoogleProvider(api_key=api_key)
        sub_name = "/".join(parts[1:])
        assert base_url.find("google") != -1, "for fool proof."
        assert base_url.find("localhost") == -1, "for fool proof."
        return GoogleModel(sub_name, provider=provider, settings=model_settings)

    elif parts[0] in {"openai", "ollama"}:
        # For Google, `openai` or in local LLM, you must pass the url. 
        assert base_url, "for fool proof."
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)
        sub_name = "/".join(parts[1:])
        return OpenAIChatModel(sub_name, provider=provider, settings=model_settings)

    else:
        from pydantic_ai_litellm import LiteLLMModel
        return LiteLLMModel(model_name=model_name, api_key=api_key, api_base=base_url, settings=model_settings)

    
class PytoyAgent:
    def __init__(self, connection: str | Connection,
                       tools: Sequence[LLMTool | Callable] = (),
                       llm_config: LLMConfig | None = None) -> None:
        llm_config = llm_config or LLMConfig()
        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)
        self._connection = connection
        self._tools = [self._normalize_tool(tool) for tool in tools]
        self._llm_config = llm_config
        
    def _normalize_tool(self, tool: LLMTool | Callable) -> Callable:
        if isinstance(tool, LLMTool):
            return tool.to_pydantic_tool()
        else:
            return tool
        
    def _make_model(self) -> PydanticAIModel:
        connection = self._connection
        base_url = connection.base_url
        model_name = connection.model
        api_key = connection.api_key
        model_settings = self._llm_config.to_pydantic_model_settings()
        return get_model(model_name, api_key, base_url, model_settings)
    
    def _make_agent[T: BaseModel](self, system_prompts: Sequence[str]):
        model = self._make_model()
        return Agent(model=model, system_prompt=system_prompts, tools=self._tools)


    def run_sync[T: BaseModel | str](self,
                               content: str | InputMessage | Sequence[InputMessage | str | Mapping],
                               llm_response_format: SyncOutputType,
                               result_cls: SyncResultClass | AgentRunResult[T] | None = None) -> str | T | AgentRunResult[T]:
        input_messages = InputMessage.to_messages(content)
        if result_cls is None:
            result_cls = llm_response_format

        # Remove system_prompts.
        system_prompts = [item.content for item in input_messages if item.role == "system"]
        input_messages = [item for item in input_messages if item.role != "system"] 

        last_user_index = None
        for index  in reversed(range(len(input_messages))):
            mes = input_messages[index]
            if mes.role == "user":
                last_user_index = index
                break
        if last_user_index is None:
            user_prompt = None
        else:
            user_prompt = input_messages[index].content
            input_messages = input_messages[:index] + input_messages[index + 1:]
            
        def _convert(message: InputMessage): 
            if message.role == "user":
                return  UserPromptPart(content=message.content)
            elif message.role == "assistant":
                return  ModelResponse(parts=[TextPart(content=message.content)])
            else:
                raise ValueError(f"`{message=}` is invalid.")
        history = [_convert(item) for item in input_messages]
        agent = self._make_agent(system_prompts=system_prompts)

        result = agent.run_sync(user_prompt=user_prompt, output_type=llm_response_format, message_history=history)

        if isinstance(result_cls, type) and issubclass(result_cls, AgentRunResult):
            return result
        elif isinstance(result_cls, type) and issubclass(result_cls, LLMOutputModel):
            return LLMOutputModel.from_pydantic_run_result(result, input_messages)
        else:
            return result.output
        



def experiment_func(name: str = DEFAULT_NAME):
    class AnswerOutput(BaseModel):
        summary: str
        key_points: list[str]
        
    from pytoy_llm.models import InputMessage
    
    mes = InputMessage(role="user", content= "Are you happy?")
    config = LLMConfig(temperature=0.7, max_tokens=150)
    agent = PytoyAgent(name, llm_config=config)
    ret = agent.run_sync(input_messages=[mes], llm_response_format=AnswerOutput, result_cls=LLMOutputModel[AnswerOutput])
    print(ret)


if __name__ == "__main__":
    experiment_func()