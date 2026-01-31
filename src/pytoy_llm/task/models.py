from __future__ import annotations

import inspect
from pytoy_llm.impl import DEFAULT_NAME, completion
from pytoy_llm.models import InputMessage, LLMMessageHistory
from functools import wraps

from pydantic import BaseModel, Field


from typing import Annotated, Any, Callable, Mapping, Protocol, Sequence, Self


class LLMTaskGivenStates[T: BaseModel | str](BaseModel):
    task_input: Annotated[str | T, Field(description="The initial input to the task / first step") ]
    given_history: Annotated[LLMMessageHistory | None, Field(description="Message given_history for context")] = None

class LLMTaskMeta(BaseModel):
    name: Annotated[str, Field(description="Human-readable task name")]
    intent: Annotated[str | None, Field(description="What the overall task is intended to do")] = None
    rules: Annotated[Sequence[str] | None, Field(description="Guiding rules or constraints for this task")] = None
    description: Annotated[str | None, Field(description="Optional longer explanation of the task purpose")] = None


class LLMTaskContext[T: BaseModel | str](BaseModel):
    given_states: Annotated[LLMTaskGivenStates, Field(description="Input and given_histories for this task.")]
    task_meta: Annotated[LLMTaskMeta, Field(description="Explanation regarding") ]
    states: Annotated[Mapping[str, Any], Field(description="This is used for sharing the states for the subsequent `Invocation`")] = {}
    connection_name:  Annotated[str, Field(description="LLM Connection")] = DEFAULT_NAME
    
    @property
    def given_history(self):
        return self.given_states.given_history
    @property
    def task_input(self):
        return self.given_states.task_input

    def completion(self, input_messages: Sequence[InputMessage], output_format: type[str] | type[T], connection_name: str | None = None) -> str | T:
        connection_name = connection_name or self.connection_name
        raw_output = completion(input_messages, output_format, connection=connection_name) #type: ignore
        if isinstance(output_format, type) and issubclass(output_format, BaseModel):
            return output_format.model_validate(raw_output)
        elif output_format is str:
            return raw_output  # type: ignore
        else:
            raise TypeError(f"Unsupported output_format type: {output_format}")


type InvocationCallable[T: BaseModel] = Callable[[Any, LLMTaskContext], T | str]

class InvocationSpec[T: BaseModel](BaseModel):
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
    spec_selector: Annotated[InvocationSpec[InvocationSpec[Any]],  
        Field(description="Function that selects which InvocationSpec to invoke based on the input")
    ]
    def invoke(self, input: Any, task_context: LLMTaskContext, /) -> Any:
        invocation_spec = self.spec_selector.invoke(input, task_context)
        return invocation_spec.invoke(input, task_context)


class LLMInvocationSpec[T: BaseModel | str](BaseModel):

    output_spec: Annotated[type[str] | type[T], Field(description="Expected type of the output from LLM")]
    create_messages: Annotated[
        Callable[[Any, LLMTaskContext[T]], Sequence[InputMessage]],
        Field(description="Function to generate the messages for LLM based on input and task context")
    ]
    connection_name:  Annotated[str | None, Field(description="LLM Connection")] = None

    def invoke(self, input: Any, task_context: LLMTaskContext[T]) -> str | T:
        input_messages = self.create_messages(input, task_context)
        return task_context.completion(input_messages, output_format=self.output_spec, connection_name=self.connection_name)
    

class LLMTaskSpec[S: BaseModel | str](BaseModel):
    """
    Represents a higher-level Task composed of multiple InvocationSpecs.

    Each step may be a direct LLMInvocationSpec or a conditional IfCondition.
    """
    invocation_specs: Annotated[
        Sequence[InvocationSpec | LLMInvocationSpec],
        Field(description="Ordered list of steps or conditional branches")
    ]
    output_spec: Annotated[type[S], Field(description="Output specification of Task.")] = str
    task_meta: Annotated[LLMTaskMeta, Field(description="Meta data for the task.")]

    def run[T: BaseModel](self,
                         task_input: T | str, history: LLMMessageHistory | None = None) -> str | S:
        given_states = LLMTaskGivenStates(given_history=history, task_input=task_input)
        task_context = LLMTaskContext(task_meta=self.task_meta, given_states=given_states)
        input = task_input
        for invocation_spec in self.invocation_specs:
            output = invocation_spec.invoke(input, task_context)
            input = output
        return output

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

