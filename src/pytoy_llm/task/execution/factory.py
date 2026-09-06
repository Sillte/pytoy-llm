from threading import RLock, Thread

from pytoy_llm.task import TaskRequest
from pytoy_llm.task.models import TaskContextState
from pytoy_llm.task.models.exceptions import TaskUnknownException
from pytoy_llm.task.shared.event import EventEmitter
from pytoy_llm.task.shared.outcome import Error

from .models import TaskExecutionExit
from .task_execution import TaskExecution


class TaskExecutionFactory:
    def __init__(self):
        pass

    def create[T](self, request: TaskRequest[T]) -> TaskExecution[T]:

        task_input = request.input
        context_state = request.context_state or TaskContextState()

        lock = RLock()
        exit_emitter = EventEmitter()

        def _main() -> TaskExecutionExit[T]:
            try:
                outcome = request.spec.run(
                    task_input=task_input, context_state=context_state, activity_sink=request.activity_sink
                )
            except Exception as e:
                outcome = Error(TaskUnknownException(e))

            exit_entity = TaskExecutionExit(id=request.id, outcome=outcome)
            with lock:
                exit_emitter.fire(exit_entity)
            return exit_entity

        thread = Thread(target=_main)
        execution = TaskExecution(thread=thread, lock=lock, request=request, exit_emitter=exit_emitter, id=request.id)

        return execution
