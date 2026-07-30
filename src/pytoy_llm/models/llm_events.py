import time
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from pytoy_llm.models.llm_metas import LLMTokens


class LLMMinimumEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the event in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    message: Annotated[str, Field(description="Short Message for the reason")] = ""
    extra: Annotated[Any, Field(description="Something which should be mentioned.")] = ""
    event_type: Annotated[str, Field(description="Event type of the event.")] = "minimum_event"


class ToolCallEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the event in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    call_id: Annotated[str | None, Field(description="Call ID of Request")] = None
    tool_name: Annotated[str | None, Field(description="The name of tool")] = None
    args: Annotated[Any, Field(description="The arguments of the call")] = {}

    event_type: Annotated[str, Field(description="Event type of the event.")] = "tool_call"


class ToolResultEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the event in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    call_id: Annotated[str | None, Field(description="Call ID of Request")] = None
    tool_name: Annotated[str | None, Field(description="The name of tool")] = None
    result: Annotated[Any, Field(description="The result of the call")] = None

    event_type: Annotated[str, Field(description="Event type of the event.")] = "tool_result"


class LLMRequestEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the event in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    call_id: Annotated[str | None, Field(description="Call ID of Request")] = None
    messages: Annotated[list[dict[str, str]], Field(description="Messages sent to the LLM.")]
    timeout: Annotated[float | None, Field(description="Timeout")] = None
    model: Annotated[str | None, Field(description="Name of model")] = None

    event_type: Annotated[str, Field(description="Event type of the event.")] = "request"


class LLMThinkingEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the event in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    content: Annotated[str | None, Field(description="Content of thinking")] = None

    event_type: Annotated[str, Field(description="Event type of the event.")] = "thinking"


class LLMResponseEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the event in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    response: Annotated[str, Field(description="Messages from the LLM.")]
    tokens: Annotated[LLMTokens | None, Field(description="Used tokens")] = None
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    event_type: Annotated[str, Field(description="Event type of the event.")] = "response"


type LLMEvent = LLMMinimumEvent | LLMRequestEvent | LLMResponseEvent | ToolResultEvent | ToolCallEvent | LLMThinkingEvent
