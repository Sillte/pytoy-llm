from typing import AsyncIterable
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

from pytoy_llm.event_sinks import EventSinkProtocol
from pytoy_llm.models.llm_events import (
    LLMEvent,
    LLMMinimumEvent,
    LLMResponseEvent,
    LLMThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)


class EventHandler:
    def __init__(self, event_sink: EventSinkProtocol) -> None:
        self._trace_id = str(uuid4())
        self._event_adapter = EventAdapter(self._trace_id)
        self._event_sink = event_sink

    async def event_stream_handler(self, ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in event_stream:
            await self.handle_event(event)

    async def handle_event(self, stream_event: AgentStreamEvent) -> None:
        match stream_event:
            case FunctionToolCallEvent():
                event = None
                # event = self._event_adapter.from_tool_call_event(stream_event)
            case FunctionToolResultEvent():
                event = None
                # event = self._event_adapter.from_tool_result_event(stream_event)
            case PartEndEvent():
                event = self._event_adapter.from_part_end_event(stream_event)
            case PartDeltaEvent() | PartStartEvent() | FinalResultEvent() | OutputToolCallEvent() | OutputToolResultEvent():
                event = None
            case _:
                event = LLMMinimumEvent(event_type="unknown_event", message=f"{stream_event.__class__.__name__}")

        if event:
            self._event_sink.emit(event)


class EventAdapter:
    def __init__(self, trace_id: str) -> None:
        self._trace_id = trace_id

    def from_tool_call_event(self, stream_event: FunctionToolCallEvent) -> ToolCallEvent:
        return ToolCallEvent(
            trace_id=self._trace_id,
            call_id=stream_event.tool_call_id,
            tool_name=stream_event.part.tool_name,
            args=stream_event.part.args,
        )

    def from_tool_result_event(self, stream_event: FunctionToolResultEvent) -> ToolResultEvent:
        return ToolResultEvent(
            trace_id=self._trace_id,
            call_id=stream_event.tool_call_id,
            tool_name=stream_event.part.tool_name,
            result=stream_event.part.content,
        )

    def from_part_end_event(self, stream_event: PartEndEvent) -> LLMEvent:
        match stream_event.part:
            case TextPart():
                event = LLMResponseEvent(
                    trace_id=self._trace_id,
                    response=stream_event.part.content,
                )

            case ThinkingPart():
                event = LLMThinkingEvent(
                    trace_id=self._trace_id,
                    content=stream_event.part.content,
                )

            case ToolCallPart():
                event = ToolCallEvent(
                    trace_id=self._trace_id, call_id=stream_event.part.tool_call_id, args=stream_event.part.args
                )

            case _:
                event = LLMMinimumEvent(
                    event_type="part_end",
                    message=f"Unsupported part: {type(stream_event.part).__name__}",
                    extra=stream_event.part,  # ignore: type
                )
        return event
