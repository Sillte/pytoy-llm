import json
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Mapping, Self, Sequence, assert_never

from pydantic import AwareDatetime, BaseModel, Field

from pytoy_llm.models.llm_activities.llm_activities import (
    LLMActivity,
    LLMRequestActivity,
    LLMResponseActivity,
    ToolCallActivity,
    ToolResultActivity,
)
from pytoy_llm.models.llm_metas import LLMTokens

type RecordDisplayMode = Literal["minimum"]


def truncate(
    target: str | list | tuple | Mapping[str, Any] | BaseModel | Any, length: int = 20
) -> str:
    suffix = "...[truncated]"
    if isinstance(target, str):
        target = target.replace("\n", "\\n")
        if length < len(target):
            return f"{target[:length]}{suffix}"
        else:
            return target
    if isinstance(target, (list, tuple)):
        return f"({' ,'.join(truncate(str(elem), length) for elem in target)})"
    if isinstance(target, Mapping):

        def _to_part(key: str, value: Any):
            return f"{key}:{truncate(str(value), length)}"

        return f"{{{' ,'.join(_to_part(key, value) for key, value in target.items())}}}"
    if isinstance(target, BaseModel):
        return truncate(target.model_dump(), length=length)
    return truncate(str(target), length=length)


class LLMActivityToolRecord(BaseModel, frozen=True):
    tool_name: Annotated[str, Field(description="Name of Tool")]
    call_id: Annotated[str, Field(description="Call ID")]
    started_at: Annotated[AwareDatetime, Field(description="Started Time")]
    finished_at: Annotated[AwareDatetime | None, Field(description="End Time")] = None
    args: Annotated[Any, Field(description="The arguments of the call")] = {}
    result: Annotated[Any, Field(description="The return of the call")] = None

    @classmethod
    def from_activities(
        cls, call_activity: ToolCallActivity, result_activity: ToolResultActivity | None = None
    ) -> Self:
        if result_activity is None:
            if call_activity.call_id is None:
                raise ValueError("`call_id` is necessary.")
            return cls(
                tool_name=(call_activity.tool_name or "NoToolName"),
                call_id=call_activity.call_id,
                started_at=datetime.fromtimestamp(call_activity.timestamp, timezone.utc),
                args=call_activity.args,
            )
        else:
            if call_activity.call_id is None:
                raise ValueError("`call_id` is necessary.")
            if call_activity.call_id != result_activity.call_id:
                raise RuntimeError("Implementation error for correspondence of activities.")

            return cls(
                tool_name=(call_activity.tool_name or result_activity.tool_name or "NoToolName"),
                call_id=call_activity.call_id,
                started_at=datetime.fromtimestamp(call_activity.timestamp, timezone.utc),
                finished_at=datetime.fromtimestamp(result_activity.timestamp, timezone.utc),
                args=call_activity.args,
                result=result_activity.result,
            )

    def to_text(self, mode: RecordDisplayMode = "minimum") -> str:
        match mode:
            case "minimum":
                try:
                    args = json.loads(self.args)
                except Exception:
                    args = self.args
                arg_text = truncate(args)
                return f"{self.tool_name}: {arg_text}"
            case _:
                assert_never(mode)
        return self.model_dump_json()


class LLMActivityRequestRecord(BaseModel, frozen=True):
    started_at: Annotated[AwareDatetime, Field(description="Started Time")]
    messages: Annotated[list[dict[str, Any]], Field(description="Messages sent to the LLM.")]
    timeout: Annotated[float | None, Field(description="Timeout")] = None
    model: Annotated[str | None, Field(description="Name of model")] = None

    @classmethod
    def from_activity(cls, request_activity: LLMRequestActivity) -> Self:
        return cls(
            started_at=datetime.fromtimestamp(request_activity.timestamp, tz=timezone.utc),
            messages=request_activity.messages,
            timeout=request_activity.timeout,
            model=request_activity.model,
        )

    def to_text(self, mode: RecordDisplayMode = "minimum") -> str:
        match mode:
            case "minimum":
                return f"RequestRecord: {self.timeout=}, {self.model=}"
            case _:
                assert_never(mode)
        return self.model_dump_json()


class LLMActivityResponseRecord(BaseModel, frozen=True):
    finished_at: Annotated[AwareDatetime, Field(description="Finished Time")]
    response: Annotated[str, Field(description="Messages from the LLM.")]
    tokens: Annotated[LLMTokens | None, Field(description="Used tokens")] = None

    @classmethod
    def from_activity(cls, response_activity: LLMResponseActivity) -> Self:
        return cls(
            finished_at=datetime.fromtimestamp(response_activity.timestamp, tz=timezone.utc),
            response=response_activity.response,
            tokens=response_activity.tokens,
        )

    def to_text(self, mode: RecordDisplayMode = "minimum") -> str:
        match mode:
            case "minimum":
                return f"ResponseRecord: {self.tokens=}"
            case _:
                assert_never(mode)
        return self.model_dump_json()


type LLMActivityRecord = (
    LLMActivityRequestRecord | LLMActivityToolRecord | LLMActivityResponseRecord
)


def time_key(activity_record: LLMActivityRecord) -> datetime:
    match activity_record:
        case LLMActivityRequestRecord():
            return activity_record.started_at
        case LLMActivityToolRecord():
            return activity_record.started_at
        case LLMActivityResponseRecord():
            return activity_record.finished_at
        case _:
            assert_never(activity_record)


class LLMActivityLog(BaseModel, frozen=True):
    records: Sequence[LLMActivityRecord]

    @classmethod
    def from_activities(cls, llm_activities: Sequence[LLMActivity]) -> Self:
        records = []

        records += [
            LLMActivityRequestRecord.from_activity(activity)
            for activity in llm_activities
            if isinstance(activity, LLMRequestActivity)
        ]

        records += [
            LLMActivityToolRecord.from_activities(call_activity, result_activity)
            for (call_activity, result_activity) in _select_tool_activities(llm_activities).values()
        ]

        records += [
            LLMActivityResponseRecord.from_activity(activity)
            for activity in llm_activities
            if isinstance(activity, LLMResponseActivity)
        ]

        return cls(records=sorted(records, key=time_key))

    def to_text(self, mode: RecordDisplayMode = "minimum") -> str:
        match mode:
            case "minimum":
                return "\n".join(record.to_text(mode) for record in self.records)
            case _:
                assert_never(mode)
        return f"ImplementaionError at {self.__class__}"


def _select_tool_activities(
    llm_activities: Sequence[LLMActivity],
) -> dict[str, tuple[ToolCallActivity, ToolResultActivity | None]]:
    tool_activities = [
        elem for elem in llm_activities if isinstance(elem, (ToolCallActivity, ToolResultActivity))
    ]
    id_to_activities: dict[str, tuple[list[ToolCallActivity], list[ToolResultActivity]]] = dict()
    for tool_activity in tool_activities:
        call_id = tool_activity.call_id
        if call_id is None:
            continue
        if call_id not in id_to_activities:
            id_to_activities[call_id] = (list(), list())
        match tool_activity:
            case ToolCallActivity():
                id_to_activities[call_id][0].append(tool_activity)
            case ToolResultActivity():
                id_to_activities[call_id][-1].append(tool_activity)
    result: dict[str, tuple[ToolCallActivity, ToolResultActivity | None]] = dict()
    for call_id, activities in id_to_activities.items():
        calls, results = activities
        if not calls:
            continue
        call_activity = min(calls, key=lambda call: call.timestamp)
        result_activity = max(results, key=lambda result: result.timestamp, default=None)
        result[call_id] = (call_activity, result_activity)
    return result
