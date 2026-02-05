from __future__ import annotations

import sys
import typer
from pydantic import BaseModel

from pytoy_llm.connection_configuration import ConnectionConfiguration, DEFAULT_NAME
from pytoy_llm.impl import completion

app = typer.Typer(
    add_completion=False,
    help="Thin, stateless LLM CLI wrapper for editor integrations.",
)


def _read_stdin() -> str:
    """Read JSON from stdin. If it's not JSON, wrap it as a simple prompt."""
    if sys.stdin.isatty():
        return "The glich occurs at CLI."
    return sys.stdin.read().strip()


@app.command()
def config(
    connection: str = typer.Option(DEFAULT_NAME, "--connection", "-c", help="Connection name"),
):
    """Making the connection file and display the path for configuration.
    It is expected that users to fill a configuration file for subsequent use.
    """
    conf_client = ConnectionConfiguration()
    path = conf_client.get_connection_path(connection)
    if path.exists():
        typer.echo(path)
        typer.echo("The above path already exist, please edit the file.")
    else:
        path = conf_client.initialize_connection_file(connection)
        typer.echo(path)
        typer.echo("The configuration is generated at the above path , please edit the file.")


@app.command()
def run(
    connection: str = typer.Option(DEFAULT_NAME, "--connection", "-c", help="Connection name"),
):
    """Execute a single stateless LLM completion."""

    input_ = _read_stdin()
    output = completion(input_, output_format="str", connection=connection)
    if isinstance(output, BaseModel):
        typer.echo(output.model_dump_json())
    else:
        typer.echo(output)


if __name__ == "__main__":
    pass
