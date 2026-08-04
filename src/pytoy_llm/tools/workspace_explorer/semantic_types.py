from typing import Annotated

from pydantic import Field

WorkspacePath = Annotated[
    str,
    Field(
        description=("Path relative to the workspace root. The path must never escape the workspace."),
        examples=["", "src", "src/pytoy_llm/api.py"],
    ),
]

FileGlob = Annotated[
    str,
    Field(
        description=("Glob pattern used to filter files. Examples: '*.py', '*.md', '*.toml'."),
        examples=["*.py"],
    ),
]

SearchPattern = Annotated[
    str,
    Field(
        description=("Text or regular expression to search for."),
        examples=["run_sync", "^class\\s+"],
    ),
]

LineNumber = Annotated[
    int,
    Field(
        ge=0,
        description=("Zero-based line number."),
        examples=[0, 42],
    ),
]

MaxResults = Annotated[
    int,
    Field(
        ge=1,
        le=1000,
        description="Maximum number of returned results.",
    ),
]

MaxDepth = Annotated[
    int,
    Field(
        ge=0,
        le=20,
        description="Maximum directory depth.",
    ),
]
