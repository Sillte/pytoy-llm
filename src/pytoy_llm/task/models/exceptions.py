from pytoy_llm.task.models.context import ExecutionContext


class InvocationException(Exception):
    def __init__(self, context: ExecutionContext, invocation_exception: Exception):
        self.context = context
        self.invocation_exception = invocation_exception

    def __str__(self):
        return f"{self.context=}\n{self.invocation_exception=}"


class TaskUnknownException(Exception):
    def __init__(self, exception: Exception):
        self.exception = exception

    def __str__(self):
        return f"{self.exception=}"


type TaskExecutionException = InvocationException | TaskUnknownException
