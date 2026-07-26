from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints, field_validator

StrictStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class Connection(BaseModel, frozen=True):
    model: Annotated[
        StrictStr,
        Field(
            description="Model Name of LLM",
            examples=["gemini/gemini-2.0-flash", "gpt-4o"],
        ),
    ]
    base_url: Annotated[
        StrictStr,
        Field(
            description="Endpoint for LLM.",
            examples=["https://"],
        ),
    ]
    api_key: Annotated[
        StrictStr,
        Field(description="Credential Information for using LLM.", examples=["SECRET-KEY"]),
    ]

    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip("/")
        else:
            return value
