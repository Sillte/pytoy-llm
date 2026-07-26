from pydantic import BaseModel

from pytoy_llm.connection_configuration import DEFAULT_NAME
from pytoy_llm.models.llm_metas import (
    LLMConfig,
)
from pytoy_llm.pydantic_agent.agent import PytoyPydanticAIAgent


def experiment_func(name: str = DEFAULT_NAME):
    class AnswerOutput(BaseModel):
        summary: str
        key_points: list[str]

    from pytoy_llm.models.llm_messages import LLMMessage

    mes = LLMMessage.chat(content="Are you happy?")
    config = LLMConfig(temperature=0.7)
    agent = PytoyPydanticAIAgent(name, llm_config=config)
    ret = agent.run(messages=mes, output_type=AnswerOutput)
    print(ret)


if __name__ == "__main__":
    experiment_func()
