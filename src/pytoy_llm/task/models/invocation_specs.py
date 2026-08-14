from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Literal, cast

from pydantic import BaseModel

from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.events.llm_events import ToolCallEvent, ToolResultEvent
from pytoy_llm.models.llm_messages import LLMMessage, LLMMessagesLike
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.models.llm_tools import LLMToolsLike
from pytoy_llm.task.models.context import (
    ContextPatch,
    ExecutionContext,
)
from pytoy_llm.task.models.invocation_results import InvocationInfo, InvocationResult, InvocationTrace
from pytoy_llm.task.models.metas import InvocationSpecMeta

type InvocationCallable[T] = Callable[[Any, ExecutionContext], T | InvocationResult[T]]


def to_invocation_result[T](
    output: T | InvocationResult[T], trace: InvocationTrace, runtime_patch: ContextPatch | None = None
) -> InvocationResult[T]:
    if isinstance(output, InvocationResult):
        return output.model_copy(update={"trace": trace, "runtime_patch": runtime_patch})
    return InvocationResult(output=output, trace=trace, runtime_patch=runtime_patch)


@dataclass(frozen=True)
class FunctionInvocationSpec[T]:
    invocator: InvocationCallable[T]
    meta: InvocationSpecMeta = field(default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A"))
    kind: Literal["function"] = "function"

    def invoke(self, input: Any, execution_context: ExecutionContext, /) -> InvocationResult:
        starttime = time.time()
        event_sink = execution_context.event_sink

        if event_sink:
            event_sink.emit(ToolCallEvent(tool_name="FunctionInvocationSpec", args=input))
        output = self.invocator(input, execution_context)
        if event_sink:
            event_sink.emit(ToolResultEvent(tool_name="FunctionInvocationSpec", result=output))

        info = InvocationInfo(started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta)
        trace = InvocationTrace(input=input, output=output, info=info)
        result = self.to_invocation_result(output, trace)
        return result

    def to_invocation_result(self, output: T | InvocationResult[T], trace: InvocationTrace) -> InvocationResult[T]:
        if isinstance(output, InvocationResult):
            return output.model_copy(update={"trace": trace})
        return InvocationResult(output=output, trace=trace)

    @classmethod
    def from_any(
        cls,
        arg: "FunctionInvocationSpec" | Callable[[Any], T] | Callable[[Any, ExecutionContext], T],
        *,
        meta: InvocationSpecMeta | None = None,
    ) -> "FunctionInvocationSpec":
        if isinstance(arg, FunctionInvocationSpec):
            if meta:
                raise ValueError("InvocationSpecMeta must not be provided when converting from InvocationSpec")
            return arg

        if meta is None:
            intent = arg.__doc__ or "an invocation function"
            name = str(arg.__name__) if hasattr(arg, "__name__") else str(arg)
            meta = InvocationSpecMeta(name=name, intent=intent.strip())

        if not callable(arg):
            raise TypeError(f"{arg} is not callable")
        sig = inspect.signature(arg)
        params = list(sig.parameters.values())

        if len(params) == 1:
            single_arg = cast(Callable[[Any], T], arg)

            @wraps(single_arg)
            def wrapped_invocator(input_data: Any, _context: ExecutionContext) -> T:
                return single_arg(input_data)

            return cls(invocator=wrapped_invocator, meta=meta)
        elif len(params) >= 2:
            arg = cast(Callable[[Any, ExecutionContext], T], arg)
            return cls(invocator=arg, meta=meta)
        else:
            raise ValueError("Callable must have at least one argument (input)")


@dataclass(frozen=True)
class SelectedInvocationSpec[T]:
    spec_selector: FunctionInvocationSpec[FunctionInvocationSpec]
    meta: InvocationSpecMeta = field(default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A"))
    kind: Literal["selector"] = "selector"

    def invoke(self, input: Any, execution_context: ExecutionContext, /) -> InvocationResult[T]:
        starttime = time.time()
        event_sink = execution_context.event_sink

        first_result = self.spec_selector.invoke(input, execution_context)
        if event_sink:
            event_sink.emit(ToolCallEvent(tool_name="SelectedInvocationSpec", args=first_result))
        spec_output = first_result.output
        second_result = spec_output.invoke(input, execution_context)
        info = InvocationInfo(started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta)
        children_traces = [first_result.trace] if first_result.trace else []
        trace = InvocationTrace(input=input, output=second_result.output, info=info, children=children_traces)
        result = to_invocation_result(second_result, trace)
        return result


@dataclass(frozen=True)
class LLMInvocationSpec[T: BaseModel | str]:
    output_type: type[T]
    create_messages: Callable[[Any, ExecutionContext], LLMMessagesLike] | Callable[[Any], LLMMessagesLike]
    llm_param: LLMParam | None = None
    connection: Connection | str | None = None
    meta: InvocationSpecMeta = field(default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A"))
    kind: Literal["llm"] = "llm"

    def invoke(self, input: Any, execution_context: ExecutionContext) -> InvocationResult[T]:
        starttime = time.time()
        if len(inspect.signature(self.create_messages).parameters) == 1:
            input_messages = self.create_messages(input)  # type:ignore
        else:
            input_messages = self.create_messages(input, execution_context)  # type: ignore
        connection = self.connection or execution_context.connection
        llm_param = self.llm_param or execution_context.llm_param
        llm_facade = LLMFacade(connection=connection, llm_param=llm_param, event_sink=execution_context.event_sink)
        result = llm_facade.completion_with_result(input_messages, output_type=self.output_type)
        output = result.output

        runtime_patch = ContextPatch(llm_messages=result.messages)

        info = InvocationInfo(started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta)
        trace = InvocationTrace(input=input, output=output, info=info, details={"llm_result": result.model_dump(mode="json")})
        return to_invocation_result(output, trace, runtime_patch=runtime_patch)


@dataclass(frozen=True)
class AgentInvocationSpec[T: BaseModel | str]:
    output_type: type[T]
    create_messages: Callable[[Any, ExecutionContext], Sequence[LLMMessage]] | Callable[[Any], Sequence[LLMMessage]]
    tools: LLMToolsLike = field(default_factory=list)
    connection: Connection | str | None = None
    llm_param: LLMParam | None = None

    meta: InvocationSpecMeta = field(default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A"))
    kind: Literal["agent"] = "agent"

    def invoke(self, input: Any, execution_context: ExecutionContext) -> InvocationResult[T]:
        starttime = time.time()
        if len(inspect.signature(self.create_messages).parameters) == 1:
            input_messages = self.create_messages(input)  # type:ignore
        else:
            input_messages = self.create_messages(input, execution_context)  # type: ignore
        connection = self.connection or execution_context.connection
        llm_param = self.llm_param or execution_context.llm_param
        llm_facade = LLMFacade(connection=connection, llm_param=llm_param, event_sink=execution_context.event_sink)
        result = llm_facade.run_with_result(input_messages, output_type=self.output_type, tools=self.tools)
        output = result.output

        runtime_patch = ContextPatch(llm_messages=result.messages)

        info = InvocationInfo(started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta)
        trace = InvocationTrace(input=input, output=output, info=info, details={"llm_result": result.model_dump(mode="json")})
        return to_invocation_result(output, trace, runtime_patch=runtime_patch)


type InvocationSpec = FunctionInvocationSpec | LLMInvocationSpec | AgentInvocationSpec | SelectedInvocationSpec
