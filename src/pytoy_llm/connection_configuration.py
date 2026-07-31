import json
from pathlib import Path
from typing import Any, Literal, assert_never

from pytoy_llm.models.connections import Connection

APPNAME = "pytoy_llm"
DEFAULT_NAME = "default"


def get_configuration_folder() -> Path:
    folder = Path.home() / ".config" / "pytoy_llm"
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def _make_default_connection_dict(kind: Literal["google", "local"] | None = None) -> dict[str, Any]:
    kind = kind or "google"
    result = {}

    match kind:
        case "google":
            result["model"] = "gemini/gemini-2.5-flash"
            result["base_url"] = "https://generativelanguage.googleapis.com/v1beta"
            result["api_key"] = "SECRET"
            result["llm_param"] = {"temperature": None}
        case "local":
            result["model"] = "ollama/qwen2.5:7b"
            result["base_url"] = "http://localhost:11434/"
            result["api_key"] = "SECRET"
            result["llm_param"] = {"temperature": None}
        case _:
            assert_never(kind)
    return result


class IllegalConfigurationError(Exception):
    """Configuration is not valid.
    The most typical case is the configuration file is generated,
    but the file is not property configured.
    """


class ConnectionConfiguration:
    """Client for handing Configuration Information"""

    def __init__(
        self,
    ) -> None:
        pass

    def initialize_connection_file(self, name: str = DEFAULT_NAME) -> Path:
        json_dict = _make_default_connection_dict("local")
        path = self.get_connection_path(name)
        path.write_text(json.dumps(json_dict, indent=4))
        return path

    def get_connection(self, name: str = DEFAULT_NAME) -> Connection:
        path = self.get_connection_path(name)
        if not path.exists():
            raise IllegalConfigurationError(f"`{name}`'s configuration file is not existent. See {path}.")
        try:
            return Connection.model_validate_json(path.read_text())
        except Exception:
            raise IllegalConfigurationError(f"`{name}`'s configuration is not valid. See `{path}`")

    def get_connection_path(self, name: str = DEFAULT_NAME) -> Path:
        root_folder = get_configuration_folder()
        connections_folder = root_folder / "connections"
        connections_folder.mkdir(exist_ok=True, parents=True)
        return connections_folder / f"{name}.json"


if __name__ == "__main__":
    ConnectionConfiguration().initialize_connection_file()
    pass
