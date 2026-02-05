import json
from pathlib import Path
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from pytoy_llm.models import Connection


APPNAME = "pytoy_llm"
DEFAULT_NAME = "default"


def get_configuration_folder() -> Path:
    folder = Path.home() / ".config" / "pytoy_llm"
    folder.mkdir(exist_ok=True, parents=True)
    return folder


def _make_default_json(model: type[BaseModel]) -> str:
    data = {}

    for field_name, field_info in model.model_fields.items():
        if field_info.default is not PydanticUndefined:
            data[field_name] = field_info.default
        # 2. default_factory（listやdictなど）がある場合
        elif field_info.default_factory is not None:
            data[field_name] = field_info.default_factory()
        else:
            # 型ヒントを取得
            field_type = field_info.annotation

            if field_type is str:
                data[field_name] = ""
            elif field_type is int or field_type is float:
                data[field_name] = 0
            elif field_type is bool:
                data[field_name] = False
            elif getattr(field_type, "__origin__", None) is list:
                data[field_name] = []
            elif getattr(field_type, "__origin__", None) is dict:
                data[field_name] = {}
            elif field_type and issubclass(field_type, BaseModel):
                data[field_name] = {}
            else:
                data[field_name] = None
    return json.dumps(data, indent=4, ensure_ascii=False)


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
        json_str = _make_default_json(Connection)
        path = self.get_connection_path(name)
        path.write_text(json_str)
        return path

    def get_connection(self, name: str = DEFAULT_NAME) -> Connection:
        path = self.get_connection_path(name)
        if not path.exists():
            raise IllegalConfigurationError(
                f"`{name}`'s configuration file is not existent. See {path}."
            )
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
    pass
