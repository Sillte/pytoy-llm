from typing import Annotated

from pydantic import BaseModel, Field, JsonValue

Namespace = Annotated[
    str,
    Field(
        description=(
            "Name of the Namespace containing the target IdeaSpace. "
            "As an argument of the function,`null` is used to refer to the default Namespace."
        ),
        examples=[
            "NameA",
            "default",
        ],
    ),
]

IdeaSpacePath = Annotated[
    str,
    Field(
        description=(
            "Directory path inside the configured IdeaSpace, relative to the IdeaSpace root. "
            "Use `.` for the IdeaSpace root. "
            "The canonical form uses `/` as the separator, does not start with `./`, "
            "and does not end with `/`."
        ),
        examples=[
            ".",
            "knowledge",
            "2026/February",
            "subspace1",
            "subspace1/subsubspace2",
        ],
    ),
]


IdeaNotePath = Annotated[
    str,
    Field(
        description=(
            "Path to an IdeaNote inside the configured IdeaSpace, relative to the IdeaSpace root. "
            "The canonical form uses `/` as the separator, does not start with `./`, "
            "and must end with `.md`."
        ),
        examples=[
            "index.md",
            "knowledge/insight.md",
            "history/Japan/note.md",
        ],
    ),
]

LocalFilePath = Annotated[
    str,
    Field(
        description=(
            "Relative path to a file inside the root which is defined by scheme and namespace."
            "The path must not escape the root."
        ),
        examples=[
            "README.md",
            "src/pytoy_llm/api.py",
            "tests/test_api.py",
        ],
    ),
]


IdeaSpacePivot = Annotated[
    IdeaSpacePath,
    Field(
        description=(
            "Starting directory for an operation in this IdeaSpace. "
            "The path is relative to the IdeaSpace root. "
            "Use `.` to represent the IdeaSpace root itself. "
            "The path identifies the directory whose descendants are inspected."
        ),
        examples=[
            ".",
            "subspace",
            "subspace/subsubspace",
        ],
    ),
]


IdeaSpaceDepth = Annotated[
    int | None,
    Field(
        description=(
            "Number of descendant levels to inspect below the pivot directory. "
            "`0` inspects only entries directly contained in the pivot directory. "
            "Larger values include entries in deeper descendant directories. "
            "`null` explores all descendant levels."
        ),
        ge=0,
    ),
]


IdeaNoteBody = Annotated[
    str,
    Field(
        description="Markdown body of the IdeaNote, excluding YAML frontmatter.",
    ),
]


IdeaNoteMetadata = Annotated[
    dict[str, JsonValue],
    Field(
        description=(
            "YAML frontmatter metadata of the IdeaNote. "
            "Use an empty mapping when the note has no metadata."
        ),
    ),
]


class IdeaNoteReference(BaseModel, frozen=True):
    """A reference to an IdeaNote within a Namespace."""

    namespace: Namespace = Field(
        description="Namespace containing the referenced IdeaNote.",
    )
    idea_note_path: IdeaNotePath = Field(
        description="Path of the IdeaNote, relative to the Namespace's IdeaSpace root.",
        examples=[
            "knowledge/python.md",
            "history/Japan/note.md",
        ],
    )
