from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import wraps
from typing import Any, Literal, cast

from pydantic import BaseModel

from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models.agent_metas import UsageLimit
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_activities.llm_activities import ToolCallActivity, ToolResultActivity
from pytoy_llm.models.llm_messages import LLMMessagesLike
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.models.llm_tools import LLMToolsLike
from pytoy_llm.task.models.context import (
    ExecutionContext,
    RuntimeContextPatch,
)
from pytoy_llm.task.models.invocation_results import (
    InvocationInfo,
    InvocationResult,
    InvocationTrace,
)
from pytoy_llm.task.models.metas import InvocationSpecMeta

type InvocationCallable[T] = Callable[[Any, ExecutionContext], T | InvocationResult[T]]


def _to_invocation_meta(
    arg: Callable[..., Any], meta: InvocationSpecMeta | None
) -> InvocationSpecMeta:
    if meta is not None:
        return meta
    intent = arg.__doc__ or "an invocation function"
    name = str(arg.__name__) if hasattr(arg, "__name__") else str(arg)
    return InvocationSpecMeta(name=name, intent=intent.strip())


def _normalize_message_creator(
    arg: Callable[[Any], LLMMessagesLike] | Callable[[Any, ExecutionContext], LLMMessagesLike],
) -> Callable[[Any, ExecutionContext], LLMMessagesLike]:
    if not callable(arg):
        raise TypeError(f"{arg} is not callable")

    params = list(inspect.signature(arg).parameters.values())
    if len(params) == 1:
        single_arg = cast(Callable[[Any], LLMMessagesLike], arg)

        @wraps(single_arg)
        def wrapped_create_messages(input_data: Any, _context: ExecutionContext) -> LLMMessagesLike:
            return single_arg(input_data)

        return wrapped_create_messages
    if len(params) == 2:
        return cast(Callable[[Any, ExecutionContext], LLMMessagesLike], arg)
    raise ValueError("Callable must accept either input or input and execution context")


def to_invocation_result[T](
    output: T | InvocationResult[T],
    trace: InvocationTrace,
    runtime_patch: RuntimeContextPatch | None = None,
) -> InvocationResult[T]:
    if isinstance(output, InvocationResult):
        return replace(output, trace=trace, runtime_patch=runtime_patch)
    return InvocationResult(output=output, trace=trace, runtime_patch=runtime_patch)


@dataclass(frozen=True)
class FunctionInvocationSpec[T]:
    invocator: InvocationCallable[T]
    meta: InvocationSpecMeta = field(
        default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A")
    )
    kind: Literal["function"] = "function"

    def invoke(self, input: Any, execution_context: ExecutionContext, /) -> InvocationResult:
        starttime = time.time()
        execution_context.emitters.activity_emitter.fire(
            ToolCallActivity(tool_name="FunctionInvocationSpec", args=input)
        )
        output = self.invocator(input, execution_context)
        execution_context.emitters.activity_emitter.fire(
            ToolResultActivity(tool_name="FunctionInvocationSpec", result=output)
        )

        info = InvocationInfo(
            started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta
        )
        trace = InvocationTrace(input=input, output=output, info=info)
        result = self.to_invocation_result(output, trace)
        return result

    def to_invocation_result(
        self, output: T | InvocationResult[T], trace: InvocationTrace
    ) -> InvocationResult[T]:
        if isinstance(output, InvocationResult):
            return replace(output, trace=trace)
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
                raise ValueError(
                    "InvocationSpecMeta must not be provided when converting from InvocationSpec"
                )
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
    meta: InvocationSpecMeta = field(
        default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A")
    )
    kind: Literal["selector"] = "selector"

    def invoke(self, input: Any, execution_context: ExecutionContext, /) -> InvocationResult[T]:
        starttime = time.time()
        first_result = self.spec_selector.invoke(input, execution_context)
        execution_context.emitters.activity_emitter.fire(
            ToolCallActivity(tool_name="SelectedInvocationSpec", args=first_result)
        )
        spec_output = first_result.output
        second_result = spec_output.invoke(input, execution_context)
        info = InvocationInfo(
            started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta
        )
        children_traces = [first_result.trace] if first_result.trace else []
        trace = InvocationTrace(
            input=input, output=second_result.output, info=info, children=children_traces
        )
        result = to_invocation_result(second_result, trace)
        return result


@dataclass(frozen=True)
class LLMInvocationSpec[T: BaseModel | str]:
    output_type: type[T]
    create_messages: (
        Callable[[Any, ExecutionContext], LLMMessagesLike] | Callable[[Any], LLMMessagesLike]
    )
    llm_param: LLMParam | None = None
    connection: Connection | str | None = None
    meta: InvocationSpecMeta = field(
        default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A")
    )
    kind: Literal["llm"] = "llm"

    @classmethod
    def from_any(
        cls,
        arg: "LLMInvocationSpec[T]"
        | Callable[[Any], LLMMessagesLike]
        | Callable[[Any, ExecutionContext], LLMMessagesLike],
        *,
        output_type: type[T] | None = None,
        llm_param: LLMParam | None = None,
        connection: Connection | str | None = None,
        meta: InvocationSpecMeta | None = None,
    ) -> "LLMInvocationSpec[T]":
        if isinstance(arg, LLMInvocationSpec):
            if any(value is not None for value in (output_type, llm_param, connection, meta)):
                raise ValueError(
                    "Configuration must not be provided when converting from InvocationSpec"
                )
            return arg
        if output_type is None:
            raise TypeError("output_type must be provided when creating an LLMInvocationSpec")
        if not callable(arg):
            raise TypeError(f"{arg} is not callable")
        return cls(
            output_type=output_type,
            create_messages=_normalize_message_creator(arg),
            llm_param=llm_param,
            connection=connection,
            meta=_to_invocation_meta(arg, meta),
        )

    def invoke(self, input: Any, execution_context: ExecutionContext) -> InvocationResult[T]:
        starttime = time.time()
        if len(inspect.signature(self.create_messages).parameters) == 1:
            input_messages = self.create_messages(input)  # type:ignore
        else:
            input_messages = self.create_messages(input, execution_context)  # type: ignore
        connection = self.connection or execution_context.connection
        llm_param = self.llm_param or execution_context.llm_param
        llm_facade = LLMFacade(
            connection=connection, llm_param=llm_param, event_emitters=execution_context.emitters
        )
        result = llm_facade.completion_with_result(input_messages, output_type=self.output_type)
        output = result.output

        runtime_patch = RuntimeContextPatch(llm_messages=result.messages)

        info = InvocationInfo(
            started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta
        )
        trace = InvocationTrace(
            input=input,
            output=output,
            info=info,
            details={"llm_result": result.model_dump(mode="json")},
        )
        return to_invocation_result(output, trace, runtime_patch=runtime_patch)


@dataclass(frozen=True)
class AgentInvocationSpec[T: BaseModel | str]:
    output_type: type[T]
    create_messages: (
        Callable[[Any, ExecutionContext], LLMMessagesLike] | Callable[[Any], LLMMessagesLike]
    )
    tools: LLMToolsLike = field(default_factory=list)
    connection: Connection | str | None = None
    llm_param: LLMParam | None = None
    usage_limit: UsageLimit | None = None

    meta: InvocationSpecMeta = field(
        default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A")
    )
    kind: Literal["agent"] = "agent"

    @classmethod
    def from_any(
        cls,
        arg: "AgentInvocationSpec[T]"
        | Callable[[Any], LLMMessagesLike]
        | Callable[[Any, ExecutionContext], LLMMessagesLike],
        *,
        output_type: type[T] | None = None,
        tools: LLMToolsLike | None = None,
        connection: Connection | str | None = None,
        llm_param: LLMParam | None = None,
        usage_limit: UsageLimit | None = None,
        meta: InvocationSpecMeta | None = None,
    ) -> "AgentInvocationSpec[T]":
        if isinstance(arg, AgentInvocationSpec):
            if any(
                value is not None
                for value in (output_type, tools, connection, llm_param, usage_limit, meta)
            ):
                raise ValueError(
                    "Configuration must not be provided when converting from InvocationSpec"
                )
            return arg
        if output_type is None:
            raise TypeError("output_type must be provided when creating an AgentInvocationSpec")
        if not callable(arg):
            raise TypeError(f"{arg} is not callable")
        return cls(
            output_type=output_type,
            create_messages=_normalize_message_creator(arg),
            tools=[] if tools is None else tools,
            connection=connection,
            llm_param=llm_param,
            usage_limit=usage_limit,
            meta=_to_invocation_meta(arg, meta),
        )

    def invoke(self, input: Any, execution_context: ExecutionContext) -> InvocationResult[T]:
        starttime = time.time()
        if len(inspect.signature(self.create_messages).parameters) == 1:
            input_messages = self.create_messages(input)  # type:ignore
        else:
            input_messages = self.create_messages(input, execution_context)  # type: ignore
        connection = self.connection or execution_context.connection
        llm_param = self.llm_param or execution_context.llm_param
        llm_facade = LLMFacade(
            connection=connection, llm_param=llm_param, event_emitters=execution_context.emitters
        )
        result = llm_facade.run_with_result(
            input_messages,
            output_type=self.output_type,
            tools=self.tools,
            usage_limit=self.usage_limit,
        )
        output = result.output

        runtime_patch = RuntimeContextPatch(llm_messages=result.messages)

        info = InvocationInfo(
            started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta
        )
        trace = InvocationTrace(
            input=input,
            output=output,
            info=info,
            details={"llm_result": result.model_dump(mode="json")},
        )
        return to_invocation_result(output, trace, runtime_patch=runtime_patch)


type InvocationSpec = (
    FunctionInvocationSpec | LLMInvocationSpec | AgentInvocationSpec | SelectedInvocationSpec
)
