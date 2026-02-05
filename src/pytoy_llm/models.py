from __future__ import annotations
import json
import inspect
from typing import Annotated, Sequence, Any, Literal, Self, cast, assert_never, get_args, get_origin, Mapping, Union
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pydantic_ai import AgentRunResult, ModelSettings
    from litellm import ModelResponse 

from pydantic import BaseModel, Field, field_validator, ConfigDict, field_validator
from pydantic import StringConstraints
from pydantic.dataclasses import dataclass

StrictStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class Connection(BaseModel, frozen=True):
    model: Annotated[
        StrictStr,
        Field(
            description="Model Name of LLM",
            examples=["gemini/gemini-2.0-flash", "gpt-4o"],
        ),
    ]
    base_url: Annotated[
        StrictStr,
        Field(
            description="Endpoint for LLM.",
            examples=["https://"],
        ),
    ]
    api_key: Annotated[
        StrictStr,
        Field(description="Credential Information for using LLM.", examples=["SECRET-KEY"]),
    ]

    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip("/")
        else:
            return value


class InputMessage(BaseModel, frozen=True):
    role: Annotated[
        Literal["system", "user", "assistant"],
        Field(description="Role of Message in LLM"),
    ]
    content: Annotated[str, Field(description="Content of the message")]
    
    @classmethod
    def from_str(cls, message: str) -> Self:
        try:
            data = json.loads(message)
            if isinstance(data, dict) and "role" in data and "content" in data:
                return cls(role=data["role"], content=data["content"])
        except Exception:
            pass
        return cls(role="user", content=message)
    
    @classmethod
    def from_any(cls, message: str | Mapping | "InputMessage") -> Self:
        if isinstance(message, str):
            return cls.from_str(message)
        elif isinstance(message, Mapping):
            return cls.model_validate(message)
        elif isinstance(message, InputMessage):
            return cls.model_validate(message)
        else:
            raise TypeError(f"Unsupported type for InputMessage: {message=}")
        
    @classmethod
    def to_messages(cls, messages: str | InputMessage | Sequence[str | Mapping | "InputMessage"]) -> Sequence["InputMessage"]:
        if isinstance(messages, str):
            return [cls.from_str(messages)]
        elif isinstance(messages, InputMessage):
            return [messages]
        elif isinstance(messages, Mapping):
            return [cls.model_validate(messages)]
        else:
            return [cls.from_any(message) for message in messages]
    

# It is uncertain whether this class is useful or not.
class StructuredMessage[T: BaseModel | str](BaseModel, frozen=True):
    role: Annotated[
        Literal["system", "user", "assistant"],
        Field(description="Role of Message in LLM"),
    ]
    content: Annotated[str | T, Field(description="Content of the message")]

class LLMMessageHistory(BaseModel, frozen=True):
    items: Annotated[Sequence[InputMessage], Field(description="")]
    
class LLMTokens(BaseModel, frozen=True):
    prompt: int
    completion: int
    total: int
    model_config = ConfigDict(extra="allow")


class LLMOutputMeta(BaseModel, frozen=True): 
    tokens: LLMTokens
    requests: int  = 1
    finish_reason: str | None = None
    raw_response: Any | None = None

    @classmethod
    def from_litellm_model_response(cls, in_response: "ModelResponse") -> Self:
        import litellm
        response = cast(litellm.TextCompletionResponse, in_response)
        usage = response.usage 
        if usage is None:
            raise ValueError("Response is strange.")
        tokens = LLMTokens(prompt=usage.prompt_tokens, completion=usage.completion_tokens, total=usage.total_tokens)
        finish_reason = response.choices[0].finish_reason
        return cls(tokens=tokens, finish_reason=finish_reason, raw_response=response, requests=1)
    @classmethod
    def from_pydantic_run_result(cls, run_result: "AgentRunResult") -> Self:
        usage = run_result.usage()
        prompt = usage.input_tokens
        completion = usage.output_tokens
        tokens = LLMTokens(prompt=prompt, completion=completion, total=prompt+completion)  # ....? really?
        return cls(tokens=tokens, finish_reason=None, requests=usage.requests,  raw_response=run_result)


class LLMOutputModel[T: BaseModel | str](BaseModel, frozen=True):
    content: Annotated[T, Field(description="The main text content from LLM")]
    meta: Annotated[LLMOutputMeta, Field(description="Meta data of the output of LLM")]
    messages: Annotated[Sequence[InputMessage],  Field(description="History of messages")]

    @classmethod
    def from_litellm_model_response(cls, response: "ModelResponse", input_messages: Sequence[InputMessage]) -> Self:
        import litellm  
        choices = cast(litellm.Choices, response.choices)
        choice = choices[0]
        content = choice.message.content
        meta = LLMOutputMeta.from_litellm_model_response(response) 
        message = cls._from_content_to_message(content)
        return cls(content=content, meta=meta, messages=[*input_messages, message])

    @classmethod
    def _from_content_to_message(cls, content: str | T) -> InputMessage:
        if isinstance(content, BaseModel):
            content = content.model_dump_json()
        else:
            content = content
        return InputMessage(role="assistant", content=content)

    @classmethod
    def from_pydantic_run_result(cls, run_result: "AgentRunResult", input_messages: Sequence[InputMessage]) -> Self:
        content: str | T = run_result.output
        meta = LLMOutputMeta.from_pydantic_run_result(run_result) 
        messages = cls._from_pydantic_messages(run_result)
        return cls(content=content, meta=meta, messages=[*input_messages, *messages])
    
    @classmethod 
    def _from_pydantic_messages(cls, run_result: "AgentRunResult") -> Sequence[InputMessage]:
        outputs = []
        for message in run_result.new_messages():
            if message.kind == "request":
                for part in message.parts:
                    if part.part_kind == "system-prompt":
                        outputs.append(InputMessage(role="system", content=part.content))
                    elif part.part_kind == "user-prompt":
                        outputs.append(InputMessage(role="user", content=str(part.content)))
            elif message.kind == "response":
                for part in message.parts:
                    if part.part_kind == "text":
                        outputs.append(InputMessage(role="assistant", content=part.content))
        outputs.append(cls._from_content_to_message(run_result.output))
        return outputs

type SyncOutputType = type[BaseModel] | type[str]
type SyncOutput = "LLMOutputModel | BaseModel | str"
type SyncResultClass = type[LLMOutputModel] | type[BaseModel]  | type[str]

class LLMConfig(BaseModel, frozen=True):
    temperature: float | None = None
    max_tokens: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    
    def to_litellm_kwargs(self) -> dict:
        return self.model_dump(exclude_none=True)

    def to_pydantic_model_settings(self) -> ModelSettings:
        from pydantic_ai import ModelSettings
        return ModelSettings(**self.model_dump(exclude_none=True))


class LLMTool(BaseModel):
    impl: Annotated[Callable, Field(description="Implementation of the tool.")]
    doc:  str | None = None
    def to_pydantic_tool(self) -> Callable:
        if self.doc is not None:
            try:
                self.impl.__doc__ = self.doc
            except Exception:
                pass
        return self.impl
    @field_validator("impl")
    def _check_callable(cls, value: Callable) -> Callable:
        def _is_allowed_type(tp: Any) -> bool:
            if isinstance(tp, type) and issubclass(tp, BaseModel):
                return True
            if tp in (str, int, float, bool):
                return True

            origin = get_origin(tp)
            if origin in (list, tuple, Sequence, Literal, Union):
                return all(_is_allowed_type(arg) for arg in get_args(tp))
            elif origin in (dict, Mapping):
                k, v = get_args(tp)
                return _is_allowed_type(k) and _is_allowed_type(v)
            elif origin is not None:
                return _is_allowed_type(origin)
            return False
        if not callable(value):
            raise TypeError(f"`{value=}` is not callable.")
        sig = inspect.signature(value)
        for param in sig.parameters.values():
            if param.annotation is inspect._empty:
                raise TypeError("Tool parameters must be type-annotated")
            if not _is_allowed_type(param.annotation):
                raise TypeError(f"Unsupported parameter type: {param.annotation}")

        if sig.return_annotation is inspect._empty:
            raise TypeError("Tool must have return type annotation")
        return value

