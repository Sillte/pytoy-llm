from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

from pytoy_llm.models.llm_metas import LLMParam

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

    api_protocol: Annotated[
        Literal["completions", "responses"] | None,
        Field(
            description="API protocol used to communicate with the LLM.",
            examples=["completions", "responses"],
        ),
    ] = None

    llm_param: Annotated[LLMParam, Field(description="Default LLM Parameter")] = LLMParam()

    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip("/")
        else:
            return value
