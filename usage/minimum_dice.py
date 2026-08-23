from typing import Annotated

from pydantic import BaseModel, Field

from pytoy_llm import run
from pytoy_llm.event_sinks import PrintEventSink


def dice(value: int) -> int:
    """It return a number by "throwing a dice."""
    return value + 3


class A(BaseModel):
    """Sentence which makes people happy."""

    dice_value: Annotated[str, Field(description="value of dice")]


# print(completion("HELLO, WORLD", output_type=A, event_sink=PrintEventSink()))
# print(completion("GOOD, WORLD", output_type=A, event_sink=PrintEventSink()))
print(
    "RES",
    run(
        "Could you please throw a dice, setting the value 1.",
        output_type=A,
        tools=[dice],
        event_sink=PrintEventSink(),
    ),
)
