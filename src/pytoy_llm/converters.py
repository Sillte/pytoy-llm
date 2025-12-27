from pytoy_llm.models import InputMessage
from pytoy_llm.models import SyncOutputMode, CustomLLMOutputModel, SyncOutput, SyncOutputType
from pydantic import ValidationError, BaseModel
from typing import Sequence, cast, Mapping
from litellm import ModelResponse, Choices


class InputConverter:
    def __init__(self):
        pass

    def to_llm_message(self, item) -> InputMessage:
        """Return one LLMMessage."""
        if isinstance(item, InputMessage):
            return item
        elif isinstance(item, dict):
            return InputMessage.model_validate(item)
        elif isinstance(item, str):
            try:
                message = InputMessage.model_validate_json(item)
            except ValidationError:
                return InputMessage(role="user", content=item)
            else:
                return message
        msg = f"{item=} cannot be converted to `LLMMessage`."
        raise ValueError(msg)

    def to_llm_messages(
        self, content: str | Sequence[InputMessage | str | Mapping]
    ) -> Sequence[InputMessage]:
        if (not isinstance(content, str)) and isinstance(content, Sequence):
            return [self.to_llm_message(elem) for elem in content]
        elif isinstance(content, str):
            return [self.to_llm_message(content)]
        msg = f"{content=} cannot be converted to `LLMMessage`."
        raise ValueError(msg)


class OutputConverter:
    def _from_str_to_type(self, output_mode: SyncOutputMode) -> SyncOutputType:
        if isinstance(output_mode, str):
            if output_mode == "all":
                output_type = ModelResponse
            elif output_mode == "str":
                output_type = str
            else:
                msg = "Unknown output_mode: {output_mode}"
                raise ValueError(msg)
        else:
            output_type = output_mode
        return output_type

    def select_response_format_argment(self, output_mode: SyncOutputMode) -> type[BaseModel] | None:
        """Return the appropriate `response_format` for `completion`.
        """
        output_type = self._from_str_to_type(output_mode)
        is_user_naive_basemodel = bool(
            issubclass(output_type, BaseModel)
            and (not issubclass(output_type, ModelResponse))
            and (not issubclass(output_type, CustomLLMOutputModel))
        )  
        if is_user_naive_basemodel:
            assert issubclass(output_type, BaseModel)
            return output_type
        return None

    def to_output(self, model_response: ModelResponse, output_mode: SyncOutputMode) -> SyncOutput:
        output_type = self._from_str_to_type(output_mode) 
        choices = cast(Choices, model_response.choices)
        if not choices:
            raise ValueError("No choices found in ModelResponse.")

        choice = choices[0]
        raw_content = choice.message.content or ""

        if output_type is str:
            return raw_content

        elif output_type == ModelResponse:
            return model_response

        elif issubclass(output_type, CustomLLMOutputModel):
            return output_type.from_litellm_model_response(model_response)
        elif issubclass(output_type, BaseModel):
            return output_type.model_validate_json(raw_content)

        msg = f"`{output_type=}` cannot be handled."
        raise ValueError(msg)
