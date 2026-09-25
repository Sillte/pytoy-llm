from typing import Any

from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_messages import LLMRequest
from pytoy_llm.models.system_prompt import SystemPrompt
from pytoy_llm.pydantic_agent import agent as agent_module
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


class RecordingAgent:
    created_system_prompt: Any = None
    run_instructions: Any = None

    def __init__(self, *, system_prompt: Any, **_: Any) -> None:
        type(self).created_system_prompt = system_prompt

    def run_sync(self, *, instructions: Any, **_: Any) -> object:
        type(self).run_instructions = instructions
        return object()


def make_agent(monkeypatch) -> PytoyPydanticAIAgent:
    monkeypatch.setattr(agent_module, "Agent", RecordingAgent)
    monkeypatch.setattr(
        agent_module.PydanticAIModelFactory,
        "create",
        lambda connection, llm_param: object(),
    )
    connection = Connection(model="openai/test", base_url="https://example.com", api_key="key")
    return PytoyPydanticAIAgent(connection)


def test_system_prompt_as_history_is_passed_to_pydantic_ai_agent(monkeypatch) -> None:
    agent = make_agent(monkeypatch)
    prompt = SystemPrompt(content="Use the history prompt.", as_history=True)

    agent._make_agent(prompt.content, tools=tuple())

    assert RecordingAgent.created_system_prompt == prompt.content


def test_system_prompt_as_instructions_is_passed_to_run(monkeypatch) -> None:
    agent = make_agent(monkeypatch)
    request = LLMRequest(
        system_prompt=SystemPrompt(content="Use the instructions.", as_history=False)
    )

    agent.run_with_native(request, output_type=str)

    assert RecordingAgent.created_system_prompt == ()
    assert RecordingAgent.run_instructions == "Use the instructions."
