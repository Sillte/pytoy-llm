from threading import RLock, Thread

from pytoy_llm.shared.event import EventEmitter
from pytoy_llm.task import TaskRequest
from pytoy_llm.task.models import ExecutionEvents, TaskContextState
from pytoy_llm.task.models.exceptions import TaskUnknownException
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
        events = ExecutionEvents()
        exit_emitter = EventEmitter()

        if request.activity_sink is not None:
            events.on_activity.event.subscribe(request.activity_sink.emit)

        def _main() -> TaskExecutionExit[T]:
            try:
                outcome = request.spec.run(
                    task_input=task_input,
                    context_state=context_state,
                    activity_sink=request.activity_sink,
                    events=events,
                )
            except Exception as e:
                outcome = Error(TaskUnknownException(e))

            exit_entity = TaskExecutionExit(id=request.id, outcome=outcome)
            with lock:
                exit_emitter.fire(exit_entity)
            return exit_entity

        thread = Thread(target=_main)
        execution = TaskExecution(
            thread=thread,
            lock=lock,
            request=request,
            events=events,
            exit_emitter=exit_emitter,
            id=request.id,
        )

        return execution
