from pydantic import BaseModel

from pytoy_llm.event_sinks import EventSinkProtocol
from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.task_request import TaskRequest
from pytoy_llm.task.models.task_response import TaskResponse


class TaskExecutor:
    def execute[T: BaseModel | str](
        self, request: TaskRequest[T], event_sink: EventSinkProtocol | None = None
    ) -> TaskResponse[T]:
        request_id = request.id
        task_input = request.input
        context_state = request.context_state or TaskContextState()
        record = request.spec.run(task_input=task_input, context_state=context_state, event_sink=event_sink)
        return TaskResponse(result=record, request_id=request_id)
