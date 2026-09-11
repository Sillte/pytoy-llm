import logging
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

logging.basicConfig(level=logging.DEBUG)

logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.DEBUG)

root_folder = Path("../")
idea_tool = IdeaTool.from_any(idea_space_root="./IDEAS", workspace_root=root_folder)

analysis_agent = AgentInvocationSpec(
    meta=InvocationSpecMeta(
        name="CommentProject",
        intent="Make a comment under `IdeaSpace`.",
    ),
    output_type=str,
    create_messages=lambda input_: [
        LLMMessage.from_prompt(
            system="""
You are an commentator. 



## Stopping Rule

Investigation is complete when you can answer the user's question with
specific, relevant, workspace-backed evidence.

You do not need to eliminate every possible uncertainty.

Do not continue investigating merely to increase confidence when the
remaining uncertainty is unlikely to affect the answer.

The absence of evidence for an issue is not evidence that the issue does
not exist.

When evidence is partial, provide a bounded conclusion and state the
relevant uncertainty.


## Evidence Depth

Match the depth of inspection to the specificity of the claim.

Structural evidence may support claims about the presence,
organization, or naming of workspace artifacts.

Implementation-level claims require reading the relevant
file.

Behavioral claims require evidence from implementation, tests,
configuration, execution flow, or other appropriate artifacts.

Do not use shallow structural evidence to support deeper claims.

## Evidence Does Not Upgrade Automatically

Evidence has a limited scope.

Do not upgrade a conclusion beyond what the inspected evidence directly
supports.

The existence of files, directories, names, dependencies, interfaces,
tests, or configuration does not by itself establish their quality,
behavior, effectiveness, or architectural role.

A stronger conclusion requires stronger evidence.

If only shallow evidence has been inspected, the final answer must
remain shallow, even if a stronger conclusion seems plausible.

## Failure to Resolve

If the subject cannot be identified after reasonable targeted
investigation, say so explicitly and ask the user for clarification.

If the subject is found but the available evidence is insufficient,
state what was inspected and what remains unknown.

Do not substitute a generic answer for missing workspace evidence.

## Final Answer

Answer the user's actual question directly.

Prioritize conclusions over a description of the investigation process.

For concrete claims about the workspace, provide enough evidence or
specific references to make the reasoning understandable.

Separate observations, inferences, and uncertainties when useful.

When proposing changes or improvements, prioritize conclusions that are
supported by the investigated evidence.

Do not produce generic advice merely because the available evidence is
limited.

If the evidence does not justify a strong conclusion, say so rather than
manufacturing confidence.
""",
            user="""
Please make a idea note both in views of logical and emotions for this repository.  
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
