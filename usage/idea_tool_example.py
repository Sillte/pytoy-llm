from pathlib import Path

from pytoy_llm.activity_sinks import PrintActivitySink
from pytoy_llm.models import UsageLimit
from pytoy_llm.models.llm_messages import LLMMessage
from pytoy_llm.task import TaskRequest, TaskSyncExecutor
from pytoy_llm.task.models import (
    AgentInvocationSpec,
)
from pytoy_llm.task.models.metas import (
    InvocationSpecMeta,
    TaskSpecMeta,
)
from pytoy_llm.task.models.task_specs import TaskSpec
from pytoy_llm.tools.idea_tool.idea_tool import IdeaTool

# logging.basicConfig(level=logging.DEBUG)

# logging.getLogger("httpx").setLevel(logging.DEBUG)
# logging.getLogger("httpcore").setLevel(logging.DEBUG)


def create_test_idea_space(path: Path):
    path.mkdir(parents=True, exist_ok=True)

    (path / "index.md").write_text(
        """\
# IdeaSpace Index

## architecture

Knowledge about the architecture and structure of the repository.

## decisions

Important design decisions and their reasons.

## experiments

Experiments, observations, and their results.

## questions

Important unanswered questions.
""",
        encoding="utf-8",
    )
    for sub_folder in ["architecture", "decisions", "experiments", "quesions"]:
        (path / sub_folder).mkdir(exist_ok=True)


root_folder = Path("../")
idea_space_root = Path("./IDEAS")
create_test_idea_space(idea_space_root)
idea_tool = IdeaTool.from_any(idea_space_root=idea_space_root, workspace_root=root_folder)

analysis_agent = AgentInvocationSpec(
    meta=InvocationSpecMeta(
        name="CommentProject",
        intent="Make a comment under `IdeaSpace`.",
    ),
    output_type=str,
    create_messages=lambda input_: [
        LLMMessage.from_prompt(
            system="""
## Rule
`index.md` may exist.
If exists, please follow the insturction of `index.md` and do not override `index.md`. 
If not, create `index.md` after investigate the `IdeaSpace` and define the insruction of this `IdeaSpace`.

## Writing Principles

Write only what is necessary for the requested purpose.
Prefer concise statements over lengthy explanations.
Do not add unnecessary context, recommendations, or background.
Do not expand a question into a proposal or analysis unless requested.
""",
            user="""
Please tell me what the user should do or ask as the next step? 
""",
        )
    ],
    tools=[idea_tool.tools],
    usage_limit=UsageLimit(max_total_tokens=2000000, max_requests=50),
)


task_spec = TaskSpec.from_specs(
    meta=TaskSpecMeta(
        name="ProjectAnalysisTask",
    ),
    invocation_specs=[
        analysis_agent,
    ],
)


request = TaskRequest(
    spec=task_spec,
    input="./",
    activity_sink=PrintActivitySink(),
)

exit = TaskSyncExecutor().execute(request)
print(exit.output)
