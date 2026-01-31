from pytoy_llm.task import LLMInvocationSpec, LLMTaskMeta, LLMTaskSpec, InvocationSpec
from pytoy_llm.task import LLMTaskExecutor, LLMTaskRequest
from pytoy_llm.models import InputMessage
from pydantic import Field, BaseModel
from typing import Annotated, Sequence


if __name__ == "__main__":
    class IncidentSummary(BaseModel):
        user_id: Annotated[str, Field(description="ID of the user involved")]
        action: Annotated[str, Field(description="What happened")]
        severity: Annotated[str, Field(description="Severity level: low / medium / high")]
        user_name: Annotated[str | None, Field(description="Readable user name, it does not appear in the log.")] = None
        
    class IncidentSummaries(BaseModel):
        items: Annotated[Sequence[IncidentSummary], Field(description="Items of incident Summary")]

    
    parse_log_invocation = LLMInvocationSpec[IncidentSummaries](
    output_spec=IncidentSummaries,
    create_messages=lambda input, ctx: [
        InputMessage(
            role="system",
            content=(
                "You are a log analysis assistant.\n"
                "Extract structured incident information from the given log.\n"
                "Follow the output schema strictly."
            )
        ),
        InputMessage(
            role="user",
            content=str(input)
        )
    ]
)
    log_input = """
    2024-01-12 09:31:22 ERROR user=U-1932 action=login_failed reason=too_many_attempts
    2024-01-12 10:05:01 WARN  user=U-2048 action=password_reset reason=suspicious_activity
    2024-01-12 11:18:45 ERROR user=U-1932 action=account_locked reason=security_policy
    """
    id_to_username = {
        "U-1932": "Alice",
        "U-2048": "Bob",
    }
    def add_user_name(summaries: IncidentSummaries, ctx) -> IncidentSummaries:
        # Immitate to access to database and add the necessary info.
        for item in summaries.items:
            item.user_name = id_to_username.get(item.user_id, "UnknownUser")
        return summaries
    user_name_invocation = InvocationSpec.from_any(add_user_name)
    

    email_invocation = LLMInvocationSpec[str](
        output_spec=str,
        create_messages=lambda summaries, ctx: [
            InputMessage(
                role="system",
                content=(
                    "You are a system notification assistant.\n"
                    "Write notification emails to users based on system incidents.\n"
                    "Write one short email per user.\n"
                    "Be polite, clear, and factual."
                )
            ),
            InputMessage(
                role="user",
                content="\n".join(
                    f"""
    User Name: {item.user_name}
    User ID: {item.user_id}
    Action: {item.action}
    Severity: {item.severity}
    """
                    for item in summaries.items
                )
            )
        ]
    )
    task_meta = LLMTaskMeta(
        name="IncidentNotificationTask",
        intent="Analyze system logs and notify affected users via email",
        rules=[
            "Do not invent incidents",
            "Do not include internal system details",
            "Write clear and polite emails"
        ]
    )
    task_spec = LLMTaskSpec[str](
        task_meta=task_meta,
        invocation_specs=[
            parse_log_invocation,   # LLM
            user_name_invocation,   # normal function
            email_invocation,       # LLM
        ],
        output_spec=str
    )
    request = LLMTaskRequest(
        task_spec=task_spec,
        task_input=log_input
    )

    response = LLMTaskExecutor().execute(request)
    print(response.output)