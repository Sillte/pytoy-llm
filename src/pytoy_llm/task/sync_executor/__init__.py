from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.task_exit import TaskExit
from pytoy_llm.task.models.task_request import TaskRequest
from pytoy_llm.task.shared.outcome import is_error, is_success


class TaskSyncExecutor:
    def execute[T](self, request: TaskRequest[T]) -> TaskExit[T]:
        task_input = request.input
        context_state = request.context_state or TaskContextState()
        outcome = request.spec.run(task_input=task_input, context_state=context_state, activity_sink=request.activity_sink)

        return TaskExit(outcome=outcome, request_id=request.id)
