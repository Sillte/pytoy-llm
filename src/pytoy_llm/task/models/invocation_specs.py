from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, Field

from pytoy_llm.llm_facade import LLMFacade
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_messages import LLMMessage, LLMMessagesLike
from pytoy_llm.models.llm_metas import LLMParam
from pytoy_llm.models.llm_tools import LLMTool
from pytoy_llm.task.models.context import (
    ContextPatch,
    ExecutionContext,
)
from pytoy_llm.task.models.invocation_results import InvocationInfo, InvocationResult, InvocationTrace
from pytoy_llm.task.models.metas import InvocationSpecMeta
from pytoy_llm.task.models.types import InvocationCallable


def to_invocation_result[T](
    output: T | InvocationResult[T], trace: InvocationTrace, runtime_patch: ContextPatch | None = None
) -> InvocationResult[T]:
    if isinstance(output, InvocationResult):
        return output.model_copy(update={"trace": trace, "runtime_patch": runtime_patch})
    return InvocationResult(output=output, trace=trace, runtime_patch=runtime_patch)


class FunctionInvocationSpec[T](BaseModel, frozen=True):
    kind: Annotated[Literal["function"], Field(description="Type of invocation")] = "function"
    meta: Annotated[InvocationSpecMeta, Field(description="Metadata about this invocation spec")]
    invocator: InvocationCallable[T]

    def invoke(self, input: Any, task_context: ExecutionContext, /) -> InvocationResult:
        starttime = time.time()
        output = self.invocator(input, task_context)

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
            # 引数2つの場合: そのまま利用
            return cls(invocator=arg, meta=meta)
        else:
            raise ValueError("Callable must have at least one argument (input)")


class SelectedInvocationSpec[T: BaseModel | str](BaseModel, frozen=True):
    kind: Annotated[Literal["selector"], Field(description="Type of invocation")] = "selector"
    meta: Annotated[InvocationSpecMeta, Field(description="Metadata about this invocation spec")]
    spec_selector: Annotated[
        FunctionInvocationSpec[FunctionInvocationSpec],
        Field(description="Function that selects which InvocationSpec to invoke based on the input"),
    ]

    def invoke(self, input: Any, task_context: ExecutionContext, /) -> InvocationResult[T]:
        starttime = time.time()
        first_result = self.spec_selector.invoke(input, task_context)
        spec_output = first_result.output
        second_result = spec_output.invoke(input, task_context)
        info = InvocationInfo(started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta)
        children_traces = [first_result.trace] if first_result.trace else []
        trace = InvocationTrace(input=input, output=second_result.output, info=info, children=children_traces)
        result = to_invocation_result(second_result, trace)
        return result


class LLMInvocationSpec[T: BaseModel | str](BaseModel):
    kind: Annotated[Literal["llm"], Field(description="Type of invocation")] = "llm"
    meta: Annotated[InvocationSpecMeta, Field(description="Metadata about this invocation spec")]

    output_type: Annotated[type[T], Field(description="Expected type of the output from LLM")]
    create_messages: Annotated[
        Callable[[Any, ExecutionContext], LLMMessagesLike] | Callable[[Any], LLMMessagesLike],
        Field(description="Function to generate the messages for LLM based on input and task context"),
    ]
    llm_param: Annotated[LLMParam | None, Field(description="LLM Parameters.")] = None
    connection: Annotated[Connection | None, Field(description="Connection for this invocation")] = None

    def invoke(self, input: Any, task_context: ExecutionContext) -> InvocationResult[T]:
        starttime = time.time()
        if len(inspect.signature(self.create_messages).parameters) == 1:
            input_messages = self.create_messages(input)  # type:ignore
        else:
            input_messages = self.create_messages(input, task_context)  # type: ignore
        connection = self.connection or task_context.connection
        llm_param = self.llm_param or task_context.llm_param
        llm_facade = LLMFacade(connection=connection, llm_param=llm_param)
        result = llm_facade.completion_with_result(input_messages, output_type=self.output_type)
        output = result.output

        runtime_patch = ContextPatch(llm_messages=result.messages)

        info = InvocationInfo(started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta)
        trace = InvocationTrace(input=input, output=output, info=info, details={"llm_result": result.model_dump()})
        return to_invocation_result(output, trace, runtime_patch=runtime_patch)


class AgentInvocationSpec[T: BaseModel | str](BaseModel):
    kind: Annotated[Literal["agent"], Field(description="Type of invocation")] = "agent"
    meta: Annotated[InvocationSpecMeta, Field(description="Metadata about this invocation spec")]
    output_type: Annotated[type[T], Field(description="Expected type of the output from LLM")]
    create_messages: Annotated[
        Callable[[Any, ExecutionContext], Sequence[LLMMessage]] | Callable[[Any], Sequence[LLMMessage]],
        Field(description="Function to generate the messages for LLM based on input and task context"),
    ]
    tools: Annotated[Sequence[Callable | LLMTool], Field(description="Tools available to the agent")] = []
    connection: Annotated[Connection | None, Field(description="LLM Connection")] = None
    llm_param: Annotated[LLMParam | None, Field(description="LLM Parameters")] = None

    def invoke(self, input: Any, task_context: ExecutionContext) -> InvocationResult[T]:
        starttime = time.time()
        if len(inspect.signature(self.create_messages).parameters) == 1:
            input_messages = self.create_messages(input)  # type:ignore
        else:
            input_messages = self.create_messages(input, task_context)  # type: ignore
        connection = self.connection or task_context.connection
        llm_param = self.llm_param or task_context.llm_param
        llm_facade = LLMFacade(connection=connection, llm_param=llm_param)
        result = llm_facade.run_with_result(input_messages, output_type=self.output_type, tools=self.tools)
        output = result.output

        runtime_patch = ContextPatch(llm_messages=result.messages)

        info = InvocationInfo(started_at=starttime, ended_at=time.time(), kind=self.kind, meta=self.meta)
        trace = InvocationTrace(input=input, output=output, info=info, details={"llm_result": result.model_dump()})
        return to_invocation_result(output, trace, runtime_patch=runtime_patch)
