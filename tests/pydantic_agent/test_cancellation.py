import asyncio
from threading import Event as ThreadEvent
from threading import Thread

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pytoy_llm.models import CancelToken, LLMCancelException
from pytoy_llm.models.connections import Connection
from pytoy_llm.models.parts import ToolCallPart
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


def test_agent_cancellation_interrupts_async_tool_and_keeps_history(monkeypatch) -> None:
    cancel_token = CancelToken()
    tool_started = ThreadEvent()
    pydantic_agent = Agent(TestModel(call_tools=["wait_for_cancellation"]))

    @pydantic_agent.tool_plain
    async def wait_for_cancellation() -> None:
        tool_started.set()
        await asyncio.Event().wait()

    agent = PytoyPydanticAIAgent(
        Connection(
            model="openai/test-model",
            base_url="https://example.com",
            api_key="test-key",
        ),
        cancel_token=cancel_token,
    )
    monkeypatch.setattr(
        agent,
        "_make_agent",
        lambda *, system_prompt, tools: pydantic_agent,
    )
    outcomes: list[BaseException | None] = []

    def run_agent() -> None:
        try:
            agent.run_with_result("hello", output_type=str)
        except BaseException as exc:
            outcomes.append(exc)
        else:
            outcomes.append(None)

    worker = Thread(target=run_agent)
    worker.start()
    try:
        assert tool_started.wait(timeout=2)
    finally:
        cancel_token.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(outcomes) == 1
    cancellation = outcomes[0]
    assert isinstance(cancellation, LLMCancelException)
    assert len(cancellation.messages) >= 2
    assert cancellation.messages[0].kind == "request"
    assert any(message.kind == "response" for message in cancellation.messages)
    assert any(
        isinstance(part, ToolCallPart)
        for message in cancellation.messages
        for part in message.parts
    )
