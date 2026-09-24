from typing import Annotated, Literal

from pydantic import BaseModel, Field

from pytoy_llm.models.llm_metas import LLMTokens


class NoExpenditure(BaseModel, frozen=True):
    kind: Literal["none"] = "none"


class LLMExpenditure(BaseModel, frozen=True):
    kind: Literal["llm"] = "llm"
    tokens: Annotated[LLMTokens, Field(description="Tokens used by the LLM invocation")]


type Expenditure = Annotated[LLMExpenditure | NoExpenditure, Field(discriminator="kind")]
