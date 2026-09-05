from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.task_request import TaskRequest
from pytoy_llm.task.models.task_response import TaskResponse


class TaskSyncExecutor:
    def execute[T](self, request: TaskRequest[T]) -> TaskResponse[T]:
        request_id = request.id
        task_input = request.input
        context_state = request.context_state or TaskContextState()
        record = request.spec.run(task_input=task_input, context_state=context_state, activity_sink=request.activity_sink)
        return TaskResponse(result=record, request_id=request_id)
