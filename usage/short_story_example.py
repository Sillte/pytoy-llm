import logging
from pathlib import Path

from pytoy_llm.activity_sinks import ActivityLogSink
from pytoy_llm.ideas.experiments.short_story import ShortStoryIdeaSpaceHandler
from pytoy_llm.task import TaskRequest, TaskSyncExecutor

logging.basicConfig(level=logging.DEBUG)

logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.DEBUG)

_this_folder = Path(__file__).absolute().parent
_SHORT_STORY = _this_folder / "SHORT_STORY"

handler = ShortStoryIdeaSpaceHandler.from_any(_SHORT_STORY)

if handler.idea_space.convention is None:
    user_prompt = """
    3つの単語は「雪」/ 「蓄電池」 / 「サンゴ礁」
    1000字程度で、何か知的な感じを与える物語。
    """
else:
    user_prompt = """
    現在の状態に応じて、作業を進めて。
    次の作業が思い浮かばない時には、
    現在の作品に対して、批評を行って。
    """
handler.assure_folder()
task_spec = handler.make_task_spec()
log_sink = ActivityLogSink()

request = TaskRequest(
    spec=task_spec,
    input=user_prompt,
    activity_sink=log_sink,
)

if handler.idea_tool:
    handler.idea_tool.mark_llm_start()

try:
    exit = TaskSyncExecutor().execute(request)
    print(exit.output)
finally:
    if handler.idea_tool:
        handler.idea_tool.mark_llm_finished()

print(log_sink.log.to_text())
