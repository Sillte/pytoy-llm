from __future__ import annotations

from pydantic import  BaseModel
from pytoy_llm.models import Connection, InputMessage, SyncOutputType, SyncResultClass, LLMOutputModel, LLMConfig
from pytoy_llm.connection_configuration import ConnectionConfiguration
from typing import Sequence, Mapping, TYPE_CHECKING, cast 
if TYPE_CHECKING:
    from litellm import ModelResponse
    
class ModelResponseConverter:
    def __init__(self, llm_response_format: "SyncOutputType | type[ModelResponse]",
                       result_cls: SyncResultClass ) -> None:
        self.llm_response_format = llm_response_format
        self.result_cls = result_cls
        
    def convert[T: BaseModel](self, response: "ModelResponse", input_messages: Sequence[InputMessage]) -> str | T | ModelResponse | LLMOutputModel:
        from litellm import Choices, ModelResponse
        choices = cast(Choices, response.choices)
        choice = choices[0]
        raw_content = choice.message.content or ""

        if issubclass(self.result_cls, str):
            return raw_content
        elif issubclass(self.result_cls, ModelResponse):
            return response
        elif issubclass(self.result_cls, LLMOutputModel):
            return LLMOutputModel.from_litellm_model_response(response, input_messages=input_messages)
        elif issubclass(self.result_cls, BaseModel):
            if not issubclass(self.llm_response_format, BaseModel):
                raise ValueError(f"`llm_response_format` must be a BaseModel when `result_cls` is a BaseModel, but got {self.llm_response_format=}")
            if isinstance(raw_content, str):
                return cast(T, self.llm_response_format.model_validate_json(raw_content))
            else:
                return cast(T, self.llm_response_format.model_validate(raw_content))
        else:
            raise ValueError(f"Invalid specification of `result_cls`, {self.result_cls=}")


class PytoyLiteLLMClient:
    """LLM Client for `vim-pytoy`.
    As you know, `vim-pytoy` is a vim(neovim/neovim+vs-code).
    Hence, only text related functions are considered.
    """

    def __init__(self, connection: str | Connection, llm_config: LLMConfig = LLMConfig()) -> None:
        if isinstance(connection, str):
            connection = ConnectionConfiguration().get_connection(connection)
        self._connection = connection
        self._llm_config = llm_config

    @property
    def connection(self) -> Connection:
        return self._connection

    def completion[T: BaseModel | str](
        self,
        content: str | InputMessage | Sequence[InputMessage | str | Mapping],
        llm_response_format: "SyncOutputType | type[ModelResponse]",
        result_cls: SyncResultClass | None = None
    ):
        from litellm import completion, ModelResponse, Choices
        if result_cls is None:
            result_cls = llm_response_format
        if isinstance(content, str) or isinstance(content, InputMessage):
            messages = [InputMessage.from_any(content)]
        elif isinstance(content, Sequence):
            messages = [InputMessage.from_any(item) for item in content]
        else:
            raise ValueError(f"`{content=}` is invalid.")

        response_format: None | type[BaseModel] = None
        if llm_response_format is str: 
            response_format = None
        else:
            response_format = llm_response_format  # type: ignore
        assert response_format is not str

        response: ModelResponse = completion(
            model=self.connection.model,
            messages=[elem.model_dump() for elem in messages],
            api_key=self.connection.api_key,
            base_url=self.connection.base_url,
            response_format=response_format,
            **self._llm_config.to_litellm_kwargs(),
        )  # type: ignore

        assert isinstance(response, ModelResponse)
        output_converter = ModelResponseConverter(llm_response_format, result_cls)
        return output_converter.convert(response, input_messages=messages)


if __name__ == "__main__":
    from pydantic import Field, BaseModel
    from typing  import Annotated
    class Target(BaseModel):
        mes: Annotated[str, Field(description="何か面白い話")]
        number: Annotated[int, Field(description="何かの素数")]


    from pytoy_llm.connection_configuration import ConnectionConfiguration, DEFAULT_NAME
    from pytoy_llm.models import InputMessage, LLMOutputModel
    import json 
    client = PytoyLiteLLMClient(DEFAULT_NAME)
    mes  = InputMessage(role="user", content=f"渡しているBaseModelの一例をください。outputはjsonで。渡すBaseModelは ```{json.dumps(Target.model_json_schema())}```")
    result = client.completion([mes], llm_response_format=str, result_cls=LLMOutputModel)
    print("result.content", result)

