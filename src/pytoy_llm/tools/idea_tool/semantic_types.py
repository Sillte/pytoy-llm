from typing import Annotated

from pydantic import Field, JsonValue

IdeaSpacePath = Annotated[
    str,
    Field(
        description=("Path inside the configured IdeaSpace. `.` means the IdeaSpace root itself."),
        examples=[".", "./knowledge", "./history/note.md"],
    ),
]

IdeaSpacePivot = Annotated[
    IdeaSpacePath,
    Field(
        description=(
            "Starting directory for an operation in this IdeaSpace. "
            "The path is relative to the IdeaSpace root. "
            "Use '.' to represent the IdeaSpace root itself. "
            "Use a subdirectory such as './architecture' "
            "to restrict the operation to that directory and its descendants."
        ),
        examples=[".", "./architecture", "./history"],
    ),
]


IdeaSpaceDepth = Annotated[
    int | None,
    Field(
        description=(
            "Number of descendant levels to explore. "
            "0 returns only immediate subspaces or notes. "
            "None explores all descendant levels."
        ),
        ge=0,
    ),
]

IdeaNoteBody = Annotated[
    str,
    Field(
        description=("Markdown body of the IdeaNote, excluding YAML frontmatter."),
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

if __name__ == "__main__":
    from pydantic import BaseModel

    class T(BaseModel):
        pivot: IdeaSpacePivot

    print(T.model_json_schema())
