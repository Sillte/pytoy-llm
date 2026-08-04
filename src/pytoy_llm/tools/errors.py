from enum import Enum

from pydantic import BaseModel, Field


class ToolErrorKind(str, Enum):
    NOT_FOUND = "not_found"
    INVALID_ARGUMENT = "invalid_argument"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_LIMIT = "resource_limit"
    PARSE_ERROR = "parse_error"
    UNKNOWN = "unknown"


class ToolError(BaseModel, frozen=True):
    """
    Structured error returned by a tool.

    The agent should inspect this information
    before deciding the next action.
    """

    kind: ToolErrorKind = Field(description=("Machine-readable category of this error."), default=ToolErrorKind.UNKNOWN)
    msg: str = Field(description=("Human-readable explanation of what happened."))

    retry: bool | None = Field(
        default=None,
        description=(
            "Whether retrying the operation may succeed. "
            "True means retry is recommended. "
            "False means retry is unlikely to help. "
            "None means the tool cannot determine this."
        ),
    )
    suggestion: str | None = Field(default=None, description=("Optional guidance for the next action by tool, if any."))
