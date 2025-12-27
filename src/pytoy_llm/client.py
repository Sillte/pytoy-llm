from litellm import completion, ModelResponse
from pytoy_llm.models import Connection, InputMessage, SyncOutputMode
from pytoy_llm.converters import InputConverter, OutputConverter
from pytoy_llm.configurations import ConfigurationClient
from typing import Sequence, Mapping


class PytoyLLMClient:
    """LLM Client for `vim-pytoy`.
    As you know, `vim-pytoy` is a vim(neovim/neovim+vs-code).
    Hence, only text related functions are considered.
    """

    def __init__(self, connection: str | Connection) -> None:
        if isinstance(connection, str):
            connection = ConfigurationClient().get_connection(connection)
        self._connection = connection

    @property
    def connection(self) -> Connection:
        return self._connection

    def completion(self, content: str | Sequence[InputMessage | str | Mapping], output_mode: SyncOutputMode = str):
        messages = InputConverter().to_llm_messages(content)
        output_converter = OutputConverter()
        response_format = output_converter.select_response_format_argment(output_mode)

        response = completion(
            model=self.connection.model,
            messages=[elem.model_dump() for elem in messages],
            api_key=self.connection.api_key,
            base_url=self.connection.base_url,
            response_format=response_format,
        )
        assert isinstance(response, ModelResponse)
        return output_converter.to_output(response, output_mode)



if __name__ == "__main__":
    pass

