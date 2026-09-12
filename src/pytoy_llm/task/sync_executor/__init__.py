from pytoy_llm.models import LLMEventEmitters
from pytoy_llm.task.models.context import TaskContextState
from pytoy_llm.task.models.task_exit import TaskExit
from pytoy_llm.task.models.task_request import TaskRequest


class TaskSyncExecutor:
    def execute[T](self, request: TaskRequest[T]) -> TaskExit[T]:
        task_input = request.input
        context_state = request.context_state or TaskContextState()
        emitters = LLMEventEmitters()
        activity_sink = request.activity_sink
        disposables = []

        if activity_sink is not None:
            disposables.append(
                emitters.on_activity.subscribe(lambda activity: activity_sink.emit(activity))
            )
        try:
            outcome = request.spec.run(
                task_input=task_input,
                context_state=context_state,
                emitters=emitters,
            )

            return TaskExit(
                outcome=outcome,
                request_id=request.id,
            )
        finally:
            for disposable in disposables:
                disposable.dispose()
