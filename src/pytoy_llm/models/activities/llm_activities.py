import time
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from pytoy_llm.models.llm_metas import LLMTokens


class LLMMinimumActivity(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the activity in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    message: Annotated[str, Field(description="Short Message for the reason")] = ""
    extra: Annotated[Any, Field(description="Something which should be mentioned.")] = ""
    activity_type: Annotated[str, Field(description="Type of the LLM activity.")] = "minimum_activity"


class ToolCallActivity(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the activity in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    call_id: Annotated[str | None, Field(description="Call ID of Request")] = None
    tool_name: Annotated[str | None, Field(description="The name of tool")] = None
    args: Annotated[Any, Field(description="The arguments of the call")] = {}

    activity_type: Annotated[str, Field(description="Type of the LLM activity.")] = "tool_call_activity"


class ToolResultActivity(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the activity in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    call_id: Annotated[str | None, Field(description="Call ID of Request")] = None
    tool_name: Annotated[str | None, Field(description="The name of tool")] = None
    result: Annotated[Any, Field(description="The result of the call")] = None

    activity_type: Annotated[str, Field(description="Type of the LLM activity.")] = "tool_result_activity"


class LLMRequestActivity(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the activity in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    call_id: Annotated[str | None, Field(description="Call ID of Request")] = None
    messages: Annotated[list[dict[str, Any]], Field(description="Messages sent to the LLM.")]
    timeout: Annotated[float | None, Field(description="Timeout")] = None
    model: Annotated[str | None, Field(description="Name of model")] = None

    activity_type: Annotated[str, Field(description="Type of the LLM activity.")] = "request_activity"


class LLMThinkingActivity(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the activity in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    content: Annotated[str | None, Field(description="Content of thinking")] = None

    activity_type: Annotated[str, Field(description="Type of the LLM activity.")] = "thinking_activity"


class LLMResponseActivity(BaseModel, frozen=True):
    model_config = ConfigDict(extra="allow")
    timestamp: Annotated[float, Field(description="Timestamp of the activity in seconds since epoch.")] = Field(
        default_factory=time.time
    )
    response: Annotated[str, Field(description="Messages from the LLM.")]
    tokens: Annotated[LLMTokens | None, Field(description="Used tokens")] = None
    trace_id: Annotated[str | None, Field(description="Trace ID of Request")] = None
    activity_type: Annotated[str, Field(description="Type of the LLM activity.")] = "response_activity"


type LLMActivity = (
    LLMMinimumActivity | LLMRequestActivity | LLMResponseActivity | ToolResultActivity | ToolCallActivity | LLMThinkingActivity
)
