from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pytoy_llm.configurations import ConfigurationClient, DEFAULT_NAME


def get_model(model_name: str, api_key: str, base_url: str):
    parts = model_name.split("/")
    if len(parts) < 1:
        raise ValueError(f"Invalid model name {model_name}")

    if parts[0] == "gemini":
        # For Google, `base_url` must not be passed.
        provider = GoogleProvider(api_key=api_key)
        sub_name = "/".join(parts[1:])
        assert base_url.find("google") != -1, "for fool proof."
        assert base_url.find("localhost") == -1, "for fool proof."
        return GoogleModel(sub_name, provider=provider)

    elif parts[0] in {"openai", "ollama"}:
        # For Google, `openai` or in local LLM, you must pass the url. 
        assert base_url, "for fool proof."
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)
        sub_name = "/".join(parts[1:])
        return OpenAIChatModel(sub_name, provider=provider)

    else:
        from pydantic_ai_litellm import LiteLLMModel
        return LiteLLMModel(model_name=model_name, api_key=api_key, api_base=base_url)


def experiment_func(name: str = DEFAULT_NAME):
    class AnswerOutput(BaseModel):
        summary: str
        key_points: list[str]

    connection = ConfigurationClient().get_connection(name)
    base_url = connection.base_url
    model_name = connection.model
    api_key = connection.api_key
    model = get_model(model_name, api_key, base_url)
    agent = Agent(model=model, output_type=AnswerOutput)
    prompt = "システムプロンプトの書き方として大事なこと"
    result = agent.run_sync(user_prompt=prompt)
    print(result)
    output = result.output

    print("要約:", output.summary)
    print("重要ポイント:", output.key_points)

if __name__ == "__main__":
    experiment_func()
