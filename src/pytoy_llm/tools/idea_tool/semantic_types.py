from typing import Annotated

from pydantic import Field, JsonValue

IdeaSpacePath = Annotated[
    str,
    Field(
        description=("Path relative to `IdeaspaceRoot`. "),
        examples=[".", "./knowledge", "./history/note.md"],
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
