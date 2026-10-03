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
from pytoy_llm.models.llm_messages import LLMRequestLike
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.models.llm_tools import LLMToolsLike
from pytoy_llm.task.models.context import (
    ExecutionContext,
    RuntimeContextPatch,
)
from pytoy_llm.task.models.expenditures import LLMExpenditure
from pytoy_llm.task.models.invocation_hooks import InvocationHooks
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


def _normalize_request_creator(
    arg: Callable[[Any], LLMRequestLike] | Callable[[Any, ExecutionContext], LLMRequestLike],
) -> Callable[[Any, ExecutionContext], LLMRequestLike]:
    if not callable(arg):
        raise TypeError(f"{arg} is not callable")

    params = list(inspect.signature(arg).parameters.values())
    if len(params) == 1:
        single_arg = cast(Callable[[Any], LLMRequestLike], arg)

        @wraps(single_arg)
        def wrapped_create_request(input_data: Any, _context: ExecutionContext) -> LLMRequestLike:
            return single_arg(input_data)

        return wrapped_create_request
    if len(params) == 2:
        return cast(Callable[[Any, ExecutionContext], LLMRequestLike], arg)
    raise ValueError("Callable must accept either input or input and execution context")


def to_invocation_result[T](
    output: T | InvocationResult[T],
    runtime_patch: RuntimeContextPatch | None = None,
) -> InvocationResult[T]:
    if isinstance(output, InvocationResult):
        if runtime_patch is None:
            return output
        return replace(output, runtime_patch=runtime_patch)
    return InvocationResult(output=output, runtime_patch=runtime_patch)


def invoke_with_hooks[T](
    hooks: InvocationHooks[T],
    context: ExecutionContext,
    operation: Callable[[], InvocationTrace[T]],
) -> InvocationTrace[T]:
    if hooks.on_start:
        try:
            hooks.on_start(context)
        except Exception:
            pass

    try:
        trace = operation()
    except Exception as exc:
        if hooks.on_exception:
            try:
                hooks.on_exception(exc)
            except Exception:
                pass
        raise
    else:
        if hooks.on_result:
            try:
                hooks.on_result(trace.result)
            except Exception:
                pass
        return trace


@dataclass(frozen=True)
class FunctionInvocationSpec[T]:
    invocator: InvocationCallable[T]
    meta: InvocationSpecMeta = field(
        default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A")
    )
    hooks: InvocationHooks[T] = field(default_factory=InvocationHooks)
    kind: Literal["function"] = "function"

    def invoke(self, input: Any, execution_context: ExecutionContext, /) -> InvocationTrace[T]:
        def operation():
            starttime = time.time()
            execution_context.emitters.activity_emitter.fire(
                ToolCallActivity(tool_name="FunctionInvocationSpec", args=input)
            )
            output = self.invocator(input, execution_context)
            result = to_invocation_result(output)
            execution_context.emitters.activity_emitter.fire(
                ToolResultActivity(tool_name="FunctionInvocationSpec", result=result.output)
            )

            info = InvocationInfo(
                started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta
            )
            return InvocationTrace(input=input, result=result, info=info)

        return invoke_with_hooks(self.hooks, execution_context, operation)

    @classmethod
    def from_any(
        cls,
        invocator: Callable[[Any], T] | Callable[[Any, ExecutionContext], T],
        *,
        meta: InvocationSpecMeta | None = None,
        hooks: InvocationHooks[T] | None = None,
    ) -> "FunctionInvocationSpec":
        if meta is None:
            intent = invocator.__doc__ or "an invocation function"
            name = str(invocator.__name__) if hasattr(invocator, "__name__") else str(invocator)
            meta = InvocationSpecMeta(name=name, intent=intent.strip())

        sig = inspect.signature(invocator)
        params = list(sig.parameters.values())

        if len(params) == 1:
            single_arg = cast(Callable[[Any], T], invocator)

            @wraps(single_arg)
            def wrapped_invocator(input_data: Any, _context: ExecutionContext) -> T:
                return single_arg(input_data)

            return cls(invocator=wrapped_invocator, meta=meta, hooks=hooks or InvocationHooks())
        elif len(params) >= 2:
            invocator = cast(Callable[[Any, ExecutionContext], T], invocator)
            return cls(invocator=invocator, meta=meta, hooks=hooks or InvocationHooks())
        else:
            raise ValueError("Callable must have at least one argument (input)")


@dataclass(frozen=True)
class SelectedInvocationSpec[T]:
    spec_selector: FunctionInvocationSpec[FunctionInvocationSpec]
    meta: InvocationSpecMeta = field(
        default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A")
    )
    hooks: InvocationHooks[T] = field(default_factory=InvocationHooks)
    kind: Literal["selector"] = "selector"

    def invoke(self, input: Any, execution_context: ExecutionContext, /) -> InvocationTrace[T]:
        def operation() -> InvocationTrace[T]:
            starttime = time.time()
            selector_trace = self.spec_selector.invoke(input, execution_context)
            execution_context.emitters.activity_emitter.fire(
                ToolCallActivity(tool_name="SelectedInvocationSpec", args=selector_trace.result)
            )
            spec_output = selector_trace.result.output
            selected_trace = spec_output.invoke(input, execution_context)
            info = InvocationInfo(
                started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta
            )
            trace = InvocationTrace(
                input=input,
                result=selected_trace.result,
                info=info,
                children=[selector_trace, selected_trace],
            )
            return trace

        return invoke_with_hooks(self.hooks, execution_context, operation)


@dataclass(frozen=True)
class LLMInvocationSpec[T: BaseModel | str]:
    output_type: type[T]
    create_request: Callable[[Any, ExecutionContext], LLMRequestLike]
    llm_param: LLMParam | None = None
    connection: Connection | str | None = None
    meta: InvocationSpecMeta = field(
        default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A")
    )
    hooks: InvocationHooks[T] = field(default_factory=InvocationHooks)
    kind: Literal["llm"] = "llm"

    @classmethod
    def from_any(
        cls,
        create_request: Callable[[Any], LLMRequestLike]
        | Callable[[Any, ExecutionContext], LLMRequestLike],
        *,
        output_type: type[T] | None = None,
        llm_param: LLMParam | None = None,
        connection: Connection | str | None = None,
        meta: InvocationSpecMeta | None = None,
        hooks: InvocationHooks[T] | None = None,
    ) -> "LLMInvocationSpec[T]":
        if output_type is None:
            raise TypeError("output_type must be provided when creating an LLMInvocationSpec")
        return cls(
            output_type=output_type,
            create_request=_normalize_request_creator(create_request),
            llm_param=llm_param,
            connection=connection,
            meta=_to_invocation_meta(create_request, meta),
            hooks=hooks or InvocationHooks(),
        )

    def invoke(self, input: Any, execution_context: ExecutionContext) -> InvocationTrace[T]:
        def operation() -> InvocationTrace[T]:
            starttime = time.time()
            input_messages = self.create_request(input, execution_context)
            connection = self.connection or execution_context.connection
            llm_param = self.llm_param or execution_context.llm_param
            llm_facade = LLMFacade(
                connection=connection,
                llm_param=llm_param,
                event_emitters=execution_context.emitters,
            )
            completion_result = llm_facade.completion_with_result(
                input_messages, output_type=self.output_type
            )
            output = completion_result.output
            invocation_result = to_invocation_result(
                output, runtime_patch=RuntimeContextPatch(llm_messages=completion_result.messages)
            )
            info = InvocationInfo(
                started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta
            )
            trace = InvocationTrace(
                input=input,
                result=invocation_result,
                info=info,
                expenditure=LLMExpenditure(tokens=completion_result.meta.tokens),
                details={"llm_result": completion_result.model_dump(mode="json")},
            )
            return trace

        return invoke_with_hooks(self.hooks, execution_context, operation)


@dataclass(frozen=True)
class AgentInvocationSpec[T: BaseModel | str]:
    output_type: type[T]
    create_request: Callable[[Any, ExecutionContext], LLMRequestLike]
    tools: LLMToolsLike = field(default_factory=list)
    connection: Connection | str | None = None
    llm_param: LLMParam | None = None
    usage_limit: UsageLimit | None = None

    meta: InvocationSpecMeta = field(
        default_factory=lambda: InvocationSpecMeta(name="NoName", intent="N/A")
    )
    hooks: InvocationHooks[T] = field(default_factory=InvocationHooks)
    kind: Literal["agent"] = "agent"

    @classmethod
    def from_any(
        cls,
        create_request: Callable[[Any], LLMRequestLike]
        | Callable[[Any, ExecutionContext], LLMRequestLike],
        *,
        output_type: type[T] | None = None,
        tools: LLMToolsLike | None = None,
        connection: Connection | str | None = None,
        llm_param: LLMParam | None = None,
        usage_limit: UsageLimit | None = None,
        meta: InvocationSpecMeta | None = None,
        hooks: InvocationHooks[T] | None = None,
    ) -> "AgentInvocationSpec[T]":
        if output_type is None:
            raise TypeError("output_type must be provided when creating an AgentInvocationSpec")
        return cls(
            output_type=output_type,
            create_request=_normalize_request_creator(create_request),
            tools=[] if tools is None else tools,
            connection=connection,
            llm_param=llm_param,
            usage_limit=usage_limit,
            meta=_to_invocation_meta(create_request, meta),
            hooks=hooks or InvocationHooks(),
        )

    def invoke(self, input: Any, execution_context: ExecutionContext) -> InvocationTrace[T]:
        def operation() -> InvocationTrace[T]:
            starttime = time.time()
            input_messages = self.create_request(input, execution_context)
            connection = self.connection or execution_context.connection
            llm_param = self.llm_param or execution_context.llm_param
            llm_facade = LLMFacade(
                connection=connection,
                llm_param=llm_param,
                event_emitters=execution_context.emitters,
            )
            agent_result = llm_facade.run_with_result(
                input_messages,
                output_type=self.output_type,
                tools=self.tools,
                usage_limit=self.usage_limit,
            )
            output = agent_result.output
            invocation_result = to_invocation_result(
                output, runtime_patch=RuntimeContextPatch(llm_messages=agent_result.messages)
            )
            info = InvocationInfo(
                started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta
            )
            trace = InvocationTrace(
                input=input,
                result=invocation_result,
                info=info,
                expenditure=LLMExpenditure(tokens=agent_result.meta.tokens),
                details={"llm_result": agent_result.model_dump(mode="json")},
            )
            return trace

        return invoke_with_hooks(self.hooks, execution_context, operation)


type InvocationSpec = (
    FunctionInvocationSpec | LLMInvocationSpec | AgentInvocationSpec | SelectedInvocationSpec
)
