from dataclasses import dataclass, field

from pytoy_llm.models.activities.llm_activities import LLMActivity
from pytoy_llm.shared.event import Event, EventEmitter


@dataclass
class LLMEventEmitters:
    on_activity: EventEmitter[LLMActivity] = field(default_factory=EventEmitter)

    @property
    def activity(self) -> Event[LLMActivity]:
        return self.on_activity.event

    def emit(self, activity: LLMActivity) -> None:
        self.on_activity.fire(activity)


type LLMEventStream = LLMEventEmitters
