from pathlib import Path

from pytoy_llm.activity_sinks import ActivityLogSink
from pytoy_llm.ideas.stories.three_word_story.studio import ThreeWordStoryStudio
from pytoy_llm.task import TaskRequest, TaskSyncExecutor

# logging.basicConfig(level=logging.DEBUG)

# logging.getLogger("httpx").setLevel(logging.DEBUG)
# logging.getLogger("httpcore").setLevel(logging.DEBUG)

_this_folder = Path(__file__).absolute().parent
_SHORT_STORY = _this_folder / "SHORT_STORY"

handler = ThreeWordStoryStudio.from_any(_SHORT_STORY)

user_prompt = """
"""

task_spec = handler.make_task_spec()
log_sink = ActivityLogSink()

request = TaskRequest(
    spec=task_spec,
    input=user_prompt,
    activity_sink=log_sink,
)

exit = TaskSyncExecutor().execute(request)
print(exit.output)
print(log_sink.log.to_text())
