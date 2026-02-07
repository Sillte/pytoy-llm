from pytoy_llm.models import InputMessage, LLMConfig, LLMTool
from pytoy_llm.task.models.context import LLMTaskContext
from pytoy_llm.task.models.context_protocols import LLMTaskContextProtocol
from pytoy_llm.task.models.schemas import InvocationEffect, InvocationMeta, InvocationRecord, InvocationRecords, InvocationSpecMeta


from pydantic import BaseModel, Field


import inspect
import time
from functools import wraps
from typing import Annotated, Any, Callable, Literal, Sequence


type InvocationCallable[T: BaseModel] = Callable[[Any, LLMTaskContextProtocol[T]], InvocationEffect | T | str]


class FunctionInvocationSpec[T: BaseModel](BaseModel, frozen=True):
    kind: Annotated[Literal["function"], Field(description="Type of invocation")] = "function"
    meta: Annotated[InvocationSpecMeta, Field(description="Metadata about this invocation spec")]
    invocator: InvocationCallable[T]

    def invoke(self, input: Any, task_context: LLMTaskContextProtocol, /) -> InvocationRecords:
        starttime = time.time()
        output_or_effect = self.invocator(input, task_context)
        effect = InvocationEffect.from_any(output_or_effect)
        invocation_meta = InvocationMeta(
            started_at=starttime, ended_at=time.time(), spec_meta=self.meta, kind=self.kind
        )
        record = InvocationRecord(input=input, output=effect.output, meta=invocation_meta)
        return InvocationRecords(entries=[record], repository_updates=effect.repository_updates)


    @classmethod
    def from_any(
        cls,
        arg: "FunctionInvocationSpec" | Callable[[T], Any] | Callable[[T, LLMTaskContext], Any],
        *,
        meta: InvocationSpecMeta | None = None,
    ) -> "FunctionInvocationSpec":
        if isinstance(arg, FunctionInvocationSpec):
            if meta:
                raise ValueError("InvocationSpecMeta must not be provided when converting from InvocationSpec")
            return arg

        if meta is None:
            intent = arg.__doc__ or "an invocation function"
            meta = InvocationSpecMeta(
                name=arg.__name__,  intent=intent.strip()
            )

        if not callable(arg):
            raise TypeError(f"{arg} is not callable")
        sig = inspect.signature(arg)
        params = list(sig.parameters.values())

        if len(params) == 1:
            @wraps(arg)  # type: ignore
            def wrapped_invocator(input_data: T, _context: LLMTaskContext) -> Any:
                return arg(input_data)  # type: ignore

            return cls(invocator=wrapped_invocator, meta=meta)  # type: ignore
        elif len(params) >= 2:
            # 引数2つの場合: そのまま利用
            return cls(invocator=arg, meta=meta)  # type: ignore
        else:
            raise ValueError("Callable must have at least one argument (input)")


class SelectedInvocationSpec[T: BaseModel](BaseModel, frozen=True):
    kind: Annotated[Literal["selector"], Field(description="Type of invocation")] = "selector"
    meta: Annotated[InvocationSpecMeta, Field(description="Metadata about this invocation spec")]
    spec_selector: Annotated[
        FunctionInvocationSpec[FunctionInvocationSpec[Any]],
        Field(
            description="Function that selects which InvocationSpec to invoke based on the input"
        ),
    ]

    def invoke(self, input: Any, task_context: LLMTaskContextProtocol, /) -> InvocationRecords:
        first_records: InvocationRecords = self.spec_selector.invoke(input, task_context)
        spec_output: FunctionInvocationSpec = first_records.output
        second_records = spec_output.invoke(input, task_context)
        return first_records.updated(second_records)


class LLMInvocationSpec[T: BaseModel | str](BaseModel):
    kind: Annotated[Literal["llm"], Field(description="Type of invocation")] = "llm"
    meta: Annotated[InvocationSpecMeta, Field(description="Metadata about this invocation spec")]

    output_spec: Annotated[
        type[T], Field(description="Expected type of the output from LLM")
    ]
    create_messages: Annotated[
        Callable[[Any, LLMTaskContextProtocol[T]], Sequence[InputMessage]],
        Field(
            description="Function to generate the messages for LLM based on input and task context"
        ),
    ]
    llm_config: Annotated[
        LLMConfig | None, Field(description="LLM Configuration for this invocation")
    ] = None
    connection_name: Annotated[str | None, Field(description="LLM Connection")] = None

    def invoke(self, input: Any, task_context: LLMTaskContextProtocol[T]) -> InvocationRecords:
        starttime = time.time()
        input_messages = self.create_messages(input, task_context)
        output_or_effect = task_context.llm_facade.completion(
            input_messages,
            output_format=self.output_spec,
            connection_name=self.connection_name,
            llm_config=self.llm_config,
        )
        effect = InvocationEffect.from_any(output_or_effect)
        meta = InvocationMeta(started_at=starttime, ended_at=time.time(), spec_meta=self.meta, kind=self.kind)
        record = InvocationRecord(input=input, output=effect.output, meta=meta)
        return InvocationRecords(entries=[record], repository_updates=effect.repository_updates)


class AgentInvocationSpec[T: BaseModel | str](BaseModel):
    kind: Annotated[Literal["agent"], Field(description="Type of invocation")] = "agent"
    meta: Annotated[InvocationSpecMeta, Field(description="Metadata about this invocation spec")]
    output_spec: Annotated[
        type[str] | type[T], Field(description="Expected type of the output from LLM")
    ]
    create_messages: Annotated[
        Callable[[Any, LLMTaskContextProtocol[T]], Sequence[InputMessage]],
        Field(
            description="Function to generate the messages for LLM based on input and task context"
        ),
    ]
    tools: Annotated[
        Sequence[Callable | LLMTool], Field(description="Tools available to the agent")
    ] = []
    llm_config: Annotated[
        LLMConfig | None, Field(description="LLM Configuration for this invocation")
    ] = None
    connection_name: Annotated[str | None, Field(description="PydanticAI Connection")] = None

    def invoke(self, input: Any, task_context: LLMTaskContextProtocol[T]) -> InvocationRecords:
        starttime = time.time()
        input_messages = self.create_messages(input, task_context)
        output_or_effect = task_context.llm_facade.run_agent(
            input_messages,
            output_format=self.output_spec,
            tools=self.tools,
            connection_name=self.connection_name,
            llm_config=self.llm_config,
        )
        effect = InvocationEffect.from_any(output_or_effect)
        meta = InvocationMeta(started_at=starttime, ended_at=time.time(), spec_meta=self.meta, kind=self.kind)
        record = InvocationRecord(input=input, output=effect.output, meta=meta)
        return InvocationRecords(entries=[record], repository_updates=effect.repository_updates)