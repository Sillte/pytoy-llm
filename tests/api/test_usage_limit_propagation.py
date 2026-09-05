from unittest.mock import Mock, call

from pytoy_llm import api
from pytoy_llm.activity_sinks import ActivitySinkProtocol
from pytoy_llm.models.agent_metas import UsageLimit
from pytoy_llm.models.llm_metas import LLMParam


def test_run_forwards_all_parameters_to_agent(monkeypatch):
    messages = "hello"
    tools = (lambda value: value,)
    llm_param = LLMParam(temperature=0.2)
    connection = "test-connection"
    activity_sink = Mock(spec=ActivitySinkProtocol)
    usage_limit = UsageLimit(max_total_tokens=100, max_requests=2)

    agent = Mock()
    agent.return_value = agent
    agent.run.return_value = "result"
    monkeypatch.setattr("pytoy_llm.llm_facade.PytoyPydanticAIAgent", agent)

    result = api.run(
        messages,
        output_type=str,
        tools=tools,
        llm_param=llm_param,
        connection=connection,
        activity_sink=activity_sink,
        usage_limit=usage_limit,
    )

    assert result == "result"
    assert agent.mock_calls == [
        call(connection, llm_param=llm_param, activity_sink=activity_sink),
        call.run(messages, output_type=str, tools=tools, usage_limit=usage_limit),
    ]
