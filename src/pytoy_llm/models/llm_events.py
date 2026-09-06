from dataclasses import dataclass, field

from pytoy_llm.models.llm_activities.llm_activities import LLMActivity
from pytoy_llm.shared.event import Event, EventEmitter


@dataclass(frozen=True)
class LLMEventEmitters:
    activity_emitter: EventEmitter[LLMActivity] = field(default_factory=EventEmitter)

    @property
    def on_activity(self) -> Event[LLMActivity]:
        return self.activity_emitter.event

    def emit_activity(self, activity: LLMActivity) -> None:
        self.activity_emitter.fire(activity)
