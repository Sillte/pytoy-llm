from typing import AsyncIterable, Sequence
from uuid import uuid4

from pydantic_ai import (
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)

from pytoy_llm.models import LLMEventEmitters, LLMMessage
from pytoy_llm.models.llm_activities.llm_activities import (
    LLMActivity,
    LLMMinimumActivity,
    LLMRequestActivity,
    LLMResponseActivity,
    LLMThinkingActivity,
    ToolCallActivity,
    ToolResultActivity,
)


class EventHandler:
    def __init__(self, event_emitters: LLMEventEmitters) -> None:
        self._trace_id = str(uuid4())
        self._event_adapter = ActivityAdapter(self._trace_id)
        self._event_emitters = event_emitters

    def emit_request(self, llm_messages: Sequence[LLMMessage]) -> None:
        messages = [elem.model_dump() for elem in llm_messages]
        activity = LLMRequestActivity(trace_id=self._trace_id, messages=messages)
        self._event_emitters.emit_activity(activity)

    async def event_stream_handler(self, ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in event_stream:
            await self.handle_event(event)

    async def handle_event(self, stream_event: AgentStreamEvent) -> None:
        match stream_event:
            case FunctionToolCallEvent():
                activity = self._event_adapter.from_tool_call_event(stream_event)
            case FunctionToolResultEvent():
                activity = self._event_adapter.from_tool_result_event(stream_event)
            case PartEndEvent():
                activity = self._event_adapter.from_part_end_event(stream_event)
            case PartDeltaEvent() | PartStartEvent() | FinalResultEvent() | OutputToolCallEvent() | OutputToolResultEvent():
                activity = None
            case _:
                activity = LLMMinimumActivity(activity_type="unknown_activity", message=f"{stream_event.__class__.__name__}")

        if activity:
            self._event_emitters.emit_activity(activity)


class ActivityAdapter:
    def __init__(self, trace_id: str) -> None:
        self._trace_id = trace_id

    def from_tool_call_event(self, stream_event: FunctionToolCallEvent) -> ToolCallActivity:
        return ToolCallActivity(
            trace_id=self._trace_id,
            call_id=stream_event.tool_call_id,
            tool_name=stream_event.part.tool_name,
            args=stream_event.part.args,
        )

    def from_tool_result_event(self, stream_event: FunctionToolResultEvent) -> ToolResultActivity:
        return ToolResultActivity(
            trace_id=self._trace_id,
            call_id=stream_event.tool_call_id,
            tool_name=stream_event.part.tool_name,
            result=stream_event.part.content,
        )

    def from_part_end_event(self, stream_event: PartEndEvent) -> LLMActivity:
        match stream_event.part:
            case TextPart():
                event = LLMResponseActivity(
                    trace_id=self._trace_id,
                    response=stream_event.part.content,
                )

            case ThinkingPart():
                event = LLMThinkingActivity(
                    trace_id=self._trace_id,
                    content=stream_event.part.content,
                )

            case ToolCallPart():
                event = ToolCallActivity(
                    trace_id=self._trace_id, call_id=stream_event.part.tool_call_id, args=stream_event.part.args
                )

            case _:
                event = LLMMinimumActivity(
                    activity_type="part_end",
                    message=f"Unsupported part: {type(stream_event.part).__name__}",
                    extra=stream_event.part,  # ignore: type
                )
        return event
