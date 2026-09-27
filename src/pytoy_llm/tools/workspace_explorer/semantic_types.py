from typing import Annotated

from pydantic import Field

WorkspaceFilePath = Annotated[
    str,
    Field(
        description=(
            "Path to a file inside the configured workspace, relative to the workspace root. "
            "The path must not escape the workspace."
        ),
        examples=[
            "README.md",
            "src/pytoy_llm/api.py",
            "tests/test_api.py",
        ],
    ),
]


WorkspaceDirectoryPath = Annotated[
    str,
    Field(
        description=(
            "Path to a directory inside the configured workspace, relative to the workspace root. "
            "Use `.` for the workspace root. "
            "The path must not escape the workspace."
        ),
        examples=[
            ".",
            "src",
            "src/pytoy_llm",
        ],
    ),
]


WorkspaceDirectoryPivot = Annotated[
    WorkspaceDirectoryPath,
    Field(
        description=(
            "Starting directory for an operation in the workspace. "
            "The path is relative to the workspace root. "
            "Use `.` to represent the workspace root itself."
        ),
        examples=[
            ".",
            "src",
            "src/pytoy_llm",
        ],
    ),
]

GlobPattern = Annotated[
    str,
    Field(
        description=("Patterns use Python pathlib.Path.match() semantics, so `**` is unavailable."),
        examples=["*", "*.py", "src/*.py"],
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

MaxBytes = Annotated[
    int,
    Field(
        ge=1,
        le=10_000_000,
        description="Maximum allowed file size in bytes.",
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
