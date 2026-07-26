from pydantic_ai.models import Model as PydanticAIModel

from pytoy_llm.models.connections import Connection
from pytoy_llm.models.llm_metas import LLMConfig


class PydanticAIModelFactory:
    @staticmethod
    def create(connection: Connection, llm_config: LLMConfig) -> PydanticAIModel:
        base_url = connection.base_url
        model_name = connection.model
        api_key = connection.api_key
        model_settings = llm_config.to_pydantic_model_settings()

        parts = model_name.split("/")
        if len(parts) < 1:
            raise ValueError(f"Invalid model name {model_name}")

        if parts[0] == "gemini":
            assert base_url.find("google") != -1, "for fool proof."
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider

            # For Google, `base_url` must not be passed.
            provider = GoogleProvider(api_key=api_key)
            sub_name = "/".join(parts[1:])
            return GoogleModel(sub_name, provider=provider, settings=model_settings)
        elif parts[0] in {"openai"}:
            assert base_url, "for fool proof."
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            # For Google, `openai` or in local LLM, you must pass the url.
            provider = OpenAIProvider(api_key=api_key, base_url=base_url)
            sub_name = "/".join(parts[1:])
            return OpenAIChatModel(sub_name, provider=provider, settings=model_settings)
        else:
            assert base_url, "for fool proof."
            from pydantic_ai_litellm import LiteLLMModel

            return LiteLLMModel(model_name=model_name, api_key=api_key, api_base=base_url, settings=model_settings)
