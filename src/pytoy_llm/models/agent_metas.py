from pydantic import BaseModel


class UsageLimit(BaseModel, frozen=True):
    max_total_tokens: int | None = None
    max_requests: int | None = None
