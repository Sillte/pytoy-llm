from pytoy_llm.task import LLMInvocationSpec, LLMTaskMeta, LLMTaskSpec, InvocationSpec, AgentInvocationSpec
from pytoy_llm.task import LLMTaskExecutor, LLMTaskRequest
from pytoy_llm.models import InputMessage
from pydantic import Field, BaseModel
from typing import Annotated, Sequence, Literal


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
        import copy
        out_summaries = copy.deepcopy(summaries)
        for item in out_summaries.items:
            item.user_name = id_to_username.get(item.user_id, "UnknownUser")
        return out_summaries

    user_name_invocation = InvocationSpec.from_any(add_user_name)
    class IncidentAction(BaseModel):
        user_id: str
        user_name: str
        action: Literal["notify", "escalate", "ignore"]   # notify / escalate / ignore
        reason: str
    class IncidentActions(BaseModel):
        actions: Annotated[
            list[IncidentAction],
            Field(description="Decided actions for incidents")
        ]
    decide_action_invocation = AgentInvocationSpec[IncidentActions](
        output_spec=IncidentActions,
        create_messages=lambda summaries, ctx: [
            InputMessage(
                role="system",
                content=(
                    "You are an incident response agent.\n"
                    "Decide what action should be taken for each incident.\n"
                    "Rules:\n"
                    "- high severity → escalate\n"
                    "- medium severity → notify\n"
                    "- low severity → ignore\n"
                    "Return structured results."
                ),
            ),
            InputMessage(
                role="user",
                content="\n".join(
                    f"user={item.user_id}, severity={item.severity}, action={item.action}, user_name={item.user_name}"
                    for item in summaries.items
                ),
            ),
        ],
    )
    

    email_invocation = LLMInvocationSpec[str](
        output_spec=str,
        create_messages=lambda actions, ctx: [
            InputMessage(
                role="system",
                content=(
                    "You are a notification assistant.\n"
                    "Write emails only for actions that are 'notify' or 'escalate'."
                ),
            ),
            InputMessage(
                role="user",
                content="\n".join(
                    f"""
    User ID: {a.user_id}
    Action: {a.action}
    Reason: {a.reason}
    """
                    for a in actions.actions
                    if a.action in ("notify", "escalate")
                ),
            ),
        ],
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
            decide_action_invocation,  # Agent
            email_invocation,       # LLM
        ],
        output_spec=str
    )
    request = LLMTaskRequest(
        task_spec=task_spec,
        task_input=log_input,
    )
    response = LLMTaskExecutor().execute(request)
    print(response.output)
    
    print("invocation_history", response.result.invocation_history)