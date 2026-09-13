from pathlib import Path
from typing import Self

from pytoy_llm.idea import IdeaSpace
from pytoy_llm.models import UsageLimit
from pytoy_llm.models.llm_messages import LLMMessage
from pytoy_llm.task.models import (
    AgentInvocationSpec,
)
from pytoy_llm.task.models.metas import (
    InvocationSpecMeta,
    TaskSpecMeta,
)
from pytoy_llm.task.models.task_specs import TaskSpec
from pytoy_llm.tools.idea_tool.idea_tool import IdeaTool

SYSTEM_PROMPT = """
You create and develop short stories based on three words.

## Responsibilities

Your task is to:

1. understand the user's request;
2. understand the IdeaSpace Convention and the current state of the IdeaSpace.
    - Specifically, `get_idea_space_root_context` returns these information.
3. create or update story artifacts when appropriate.

## Instruction Sources

Instructions may come from two sources:

- User Prompt
- `dashboard.md`

A User Prompt is an instruction for the current interaction.

`dashboard.md` contains persistent intent and the current state of this IdeaSpace.

Treat the persistent intent in `dashboard.md` as user-owned.

Do not change its semantic meaning unless the user explicitly authorizes the change.

## Working Principles

Before making substantial changes:

1. read the IdeaSpace Convention;
2. read `dashboard.md` if it exists;
3. inspect relevant existing notes;
4. determine what is already known and what is uncertain.

Do not invent facts, decisions, or user requirements.

When important information is missing, ask the user rather than silently
creating requirements.

## Output and Persistence

Use the IdeaSpace to preserve information that is useful for subsequent
work or is part of the story itself.

Do not persist private reasoning or chain-of-thought.

When an action produces a meaningful artifact, store that artifact in the
appropriate location according to the IdeaSpace Convention.

Report the result of the work to the user when appropriate.
""".strip()

CONVENTION = """
# IdeaSpace Convention

This IdeaSpace is used to create and develop a short story from three words.

## Directory Structure

### `plots/`

This directory contains story plots and story development notes.

- The LLM may create and edit notes in this directory.
- Humans may create and edit notes in this directory.
- Each plot note should describe a meaningful story development.

### `outputs/`

This directory contains story outputs intended to be read as completed or presentable works.

- The LLM may create and edit notes in this directory.
- Humans may create and edit notes in this directory.
- An output should be understandable without requiring access to internal plot notes.

## Dashboard

### File

`dashboard.md`

### Purpose of `dashboard.md`

`dashboard.md` is the persistent control document of this IdeaSpace.

It records the current purpose, concept, three words, and persistent
instruction for the story.

The LLM must read `dashboard.md` before making substantial changes to the story.

### Sections

* Three Words: 3-words this short story handles and there relation in the story.  
* Concept: Description regarding the story. e.g. Why this story is interesting or intriguing. 
* Master Instrucion: Describe the persistent instruction provided by the user.
    - The `Master Instruction` represents the user's persistent intent.
    - The LLM must preserve its semantic meaning.
    - The LLM may modify this section only when the user authorizes the modification.

### Structure

`dashboard.md` should follow this structure:
    
```markdown
# Dashboard

## Purpose

### Three Words

- `<word 1>`
- `<word 2>`
- `<word 3>`

<Brief description of how the three words are connected.>

### Concept

<Description of the purpose, theme, or interesting idea of the story.>

## Master Instruction

<Persistent instruction provided by the user.>
```


## Language Selection
Use English or Japanese (日本語).

The language of generated files is determined as follows:

1. If `dashboard.md` exists, use the same language as `dashboard.md`.
2. If `dashboard.md` does not exist, use the language of the current User Prompt
   when creating it.
3. Otherwise, use English.

Once `dashboard.md` exists, its language is the default language for new IdeaSpace artifacts.

## Initialization

If `dashboard.md` does not exist, initialize it before developing the story.

When creating `dashboard.md`:

1. preserve the user's intention as much as possible;
2. do not invent requirements that are not implied by the user's request;
3. if essential information is missing:
   - if the user has authorized the LLM to decide missing details,
     make reasonable decisions;
   - otherwise, report what is missing and wait for the user's response.

## File Naming

Plot files or output files should use:

<YYYY-MM-DD-HH-MM>_<short-description>.md

"""


class ShortStoryIdeaSpaceHandler:
    def __init__(self, idea_space: IdeaSpace):
        self.idea_space = idea_space
        self._idea_tool: IdeaTool | None = None

    @classmethod
    def from_any(cls, idea_space_folder: Path | str) -> Self:
        idea_space = IdeaSpace.from_path(idea_space_folder)
        return cls(idea_space=idea_space)

    @property
    def idea_tool(self) -> IdeaTool | None:
        return self._idea_tool

    @property
    def folder_path(self) -> Path:
        return self.idea_space.folder_path

    def assure_folder(self):
        if not self.folder_path.exists():
            self.folder_path.mkdir(exist_ok=True, parents=True)
            (self.folder_path / ".convention.md").write_text(CONVENTION)

    def make_invocation_spec(
        self, usage_limit: UsageLimit | None = None
    ) -> AgentInvocationSpec[str]:
        self.assure_folder()
        usage_limit = usage_limit or UsageLimit(max_total_tokens=2000000, max_requests=50)
        self._idea_tool = IdeaTool.from_any(
            idea_space_root=self.folder_path, workspace_root=self.folder_path
        )
        return AgentInvocationSpec(
            meta=InvocationSpecMeta(
                name="CommentProject",
                intent="Make a comment under `IdeaSpace`.",
            ),
            output_type=str,
            create_messages=lambda input_: [
                LLMMessage.from_prompt(
                    system=SYSTEM_PROMPT.strip(),
                    user=input_,
                )
            ],
            tools=[self._idea_tool.tools],
            usage_limit=usage_limit,
        )

    def make_task_spec(self, usage_limit: UsageLimit | None = None):
        return TaskSpec.from_specs(
            meta=TaskSpecMeta(
                name="ProjectAnalysisTask",
            ),
            invocation_specs=[
                self.make_invocation_spec(usage_limit=usage_limit),
            ],
        )
