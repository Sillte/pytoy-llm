from __future__ import annotations

import inspect
from pytoy_llm.impl import completion
from pytoy_llm.impl import run_agent
from pytoy_llm.connection_configuration import DEFAULT_NAME
from pytoy_llm.models import InputMessage, LLMMessageHistory, LLMTool, LLMConfig
from functools import wraps

from pydantic import BaseModel, Field, PrivateAttr


from typing import Annotated, Any, Callable, Mapping,  Sequence, Self, cast, Literal


class LLMTaskGivenStates[T: BaseModel | str](BaseModel):
    task_input: Annotated[str | T, Field(description="The initial input to the task / first step") ]
    given_history: Annotated[LLMMessageHistory | None, Field(description="Message given_history for context")] = None

class LLMTaskMeta(BaseModel):
    name: Annotated[str, Field(description="Human-readable task name")]
    intent: Annotated[str | None, Field(description="What the overall task is intended to do")] = None
    rules: Annotated[Sequence[str] | None, Field(description="Guiding rules or constraints for this task")] = None
    description: Annotated[str | None, Field(description="Optional longer explanation of the task purpose")] = None
    
class StateRepository(BaseModel):
    _data: dict[str, Any] = PrivateAttr(default_factory=dict)
    
    @property
    def snapshot(self) -> dict[str, Any]:
        return self._data.copy()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

class InvocationRecord(BaseModel, frozen=True):
    invocation_type: Annotated[
        Literal["llm", "agent", "function", "selector"],
        Field(description="Type of invocation executed in this step")
    ]

    input: Annotated[
        Any,
        Field(description="Input value given to this invocation")
    ]

    output: Annotated[
        Any,
        Field(description="Output value produced by this invocation")
    ]


class LLMTaskContext[T: BaseModel | str](BaseModel):
    given_states: Annotated[LLMTaskGivenStates, Field(description="Input and given_histories for this task.")]
    task_meta: Annotated[LLMTaskMeta, Field(description="Explanation regarding") ]
    repository: Annotated[
        StateRepository,
        Field(description="This is used for sharing the states for the subsequent `Invocation`")
    ] = Field(default_factory=StateRepository)
    invocation_records: Annotated[list[InvocationRecord], Field(description="Records of invocations executed during this task.")] = []

    llm_config: Annotated[LLMConfig | None, Field(description="LLM Configuration for this invocation")] = None
    connection_name:  Annotated[str, Field(description="LLM Connection")] = DEFAULT_NAME
    
    @property
    def given_history(self):
        return self.given_states.given_history
    @property
    def task_input(self):
        return self.given_states.task_input
    
    def add_invocation_record(self, input: Any, output: Any, invocation_type: Literal["llm", "agent", "function", "selector"]) -> None:
        self.invocation_records.append(InvocationRecord(
            invocation_type=invocation_type,
            input=input,
            output=output,
        ))


    def completion(self, input_messages: Sequence[InputMessage], output_format: type[str] | type[T], llm_config: LLMConfig | None,  connection_name: str | None = None) -> str | T:
        connection_name = connection_name or self.connection_name
        llm_config = llm_config or self.llm_config
        raw_output = completion(input_messages, output_format, connection=connection_name, llm_config=llm_config) #type: ignore
        if isinstance(output_format, type) and issubclass(output_format, BaseModel):
            return output_format.model_validate(raw_output)
        elif output_format is str:
            return raw_output  # type: ignore
        else:
            raise TypeError(f"Unsupported output_format type: {output_format}")

    def run_agent(self,
                input_messages: Sequence[InputMessage],
                output_format: type[str] | type[T],
                tools: Sequence[Callable | LLMTool] = (),
                llm_config: LLMConfig | None = None,
                connection_name: str | None = None) -> str | T:
        """Alias of `run_agent` for better readability."""
        connection_name = connection_name or self.connection_name
        llm_config = llm_config or self.llm_config
        result = run_agent(input_messages, output_format, tools=tools, connection=connection_name, llm_config=llm_config)
        return cast(str | T, result)


type InvocationCallable[T: BaseModel] = Callable[[Any, LLMTaskContext], T | str]

class InvocationSpec[T: BaseModel](BaseModel):
    kind: Annotated[Literal["function"], Field(description="Type of invocation")] = "function"
    invocator: InvocationCallable[T]
    def invoke(self, input: Any, task_context: LLMTaskContext, /) -> Any:
        return self.invocator(input, task_context)
    
    @classmethod
    def from_any(cls, arg: "InvocationSpec" | Callable[[T], Any] | Callable[[T, LLMTaskContext], Any]) -> "InvocationSpec":
        if isinstance(arg, InvocationSpec):
            return arg
        if not callable(arg):
            raise TypeError(f"{arg} is not callable")
        sig = inspect.signature(arg)
        params = list(sig.parameters.values())

        if len(params) == 1:
            @wraps
            def wrapped_invocator(input_data: T, _context: LLMTaskContext) -> Any:
                return arg(input_data)  # type: ignore
            return cls(invocator=wrapped_invocator)  # type: ignore
        elif len(params) >= 2:
            # 引数2つの場合: そのまま利用
            return cls(invocator=arg)  # type: ignore
        else:
            raise ValueError("Callable must have at least one argument (input)")
        

class SelectionInvocator[T:BaseModel](BaseModel):
    kind: Annotated[Literal["selector"], Field(description="Type of invocation")] = "selector"
    spec_selector: Annotated[InvocationSpec[InvocationSpec[Any]],  
        Field(description="Function that selects which InvocationSpec to invoke based on the input")
    ]
    def invoke(self, input: Any, task_context: LLMTaskContext, /) -> Any:
        invocation_spec = self.spec_selector.invoke(input, task_context)
        return invocation_spec.invoke(input, task_context)


class LLMInvocationSpec[T: BaseModel | str](BaseModel):
    kind: Annotated[Literal["llm"], Field(description="Type of invocation")] = "llm"

    output_spec: Annotated[type[str] | type[T], Field(description="Expected type of the output from LLM")]
    create_messages: Annotated[
        Callable[[Any, LLMTaskContext[T]], Sequence[InputMessage]],
        Field(description="Function to generate the messages for LLM based on input and task context")
    ]
    llm_config: Annotated[LLMConfig | None, Field(description="LLM Configuration for this invocation")] = None
    connection_name:  Annotated[str | None, Field(description="LLM Connection")] = None

    def invoke(self, input: Any, task_context: LLMTaskContext[T]) -> str | T:
        input_messages = self.create_messages(input, task_context)
        return task_context.completion(input_messages, output_format=self.output_spec, connection_name=self.connection_name, llm_config=self.llm_config)

class AgentInvocationSpec[T:BaseModel | str](BaseModel):
    kind: Annotated[Literal["agent"], Field(description="Type of invocation")] = "agent"
    output_spec: Annotated[type[str] | type[T], Field(description="Expected type of the output from LLM")]
    create_messages: Annotated[
        Callable[[Any, LLMTaskContext[T]], Sequence[InputMessage]],
        Field(description="Function to generate the messages for LLM based on input and task context")
    ]
    tools: Annotated[Sequence[Callable | LLMTool], Field(description="Tools available to the agent")] = []
    llm_config: Annotated[LLMConfig | None, Field(description="LLM Configuration for this invocation")] = None
    connection_name:  Annotated[str | None, Field(description="PydanticAI Connection")] = None

    def invoke(self, input: Any, task_context: LLMTaskContext[T]) -> str | T:
        input_messages = self.create_messages(input, task_context)
        return task_context.run_agent(input_messages, output_format=self.output_spec, tools=self.tools, connection_name=self.connection_name, llm_config=self.llm_config)


class LLMTaskRunResult[T: BaseModel | str](BaseModel):
    task_name: Annotated[str, Field(description="Name of the executed task")]

    output: Annotated[
        T,
        Field(description="Final output produced by the task")
    ]

    invocation_history: Annotated[
        Sequence[InvocationRecord],
        Field(description="Ordered history of all invocations executed in this task")
    ]

    repository_snapshot: Annotated[
        dict[str, Any],
        Field(description="Final snapshot of the shared state repository")
    ]
    

class LLMTaskSpec[S: BaseModel | str](BaseModel):
    """
    Represents a higher-level Task composed of multiple InvocationSpecs.
    """
    invocation_specs: Annotated[
        Sequence[InvocationSpec | LLMInvocationSpec | AgentInvocationSpec],
        Field(description="Ordered list of steps or conditional branches")
    ]
    output_spec: Annotated[type[S], Field(description="Output specification of Task.")] = str
    task_meta: Annotated[LLMTaskMeta, Field(description="Meta data for the task.")]

    def run[T: BaseModel](self,
                         task_input: T | str, history: LLMMessageHistory | None = None) ->  LLMTaskRunResult[S]:
        given_states = LLMTaskGivenStates(given_history=history, task_input=task_input)
        task_context = LLMTaskContext(task_meta=self.task_meta, given_states=given_states)
        input = task_input
        for invocation_spec in self.invocation_specs:
            output = invocation_spec.invoke(input, task_context)
            task_context.add_invocation_record(input=input, output=output, invocation_type=invocation_spec.kind)
            input = output
        return LLMTaskRunResult(task_name=self.task_meta.name, output=output,
                                invocation_history=task_context.invocation_records,
                                repository_snapshot=task_context.repository._data)  # type: ignore

    @property
    def name(self) -> str:
        return self.task_meta.name
    
    @classmethod
    def from_single_spec(cls, task_meta: str | LLMTaskMeta, llm_invocation_spec: LLMInvocationSpec) -> Self:
        """Utility function for construction the task with 1 LLMInvocation. 
        """
        if isinstance(task_meta, str):
            task_meta = LLMTaskMeta(name=task_meta)
        return cls(task_meta=task_meta, invocation_specs=[llm_invocation_spec], output_spec=llm_invocation_spec.output_spec)

