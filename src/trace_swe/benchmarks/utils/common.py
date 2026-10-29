"""Shared agent utilities for benchmark runners."""

from trace_swe.agent.default import DefaultAgent
from trace_swe.agent.event_log import EventLogConfig, EventLogMixin
from trace_swe.benchmarks.utils.batch_progress import RunBatchProgressManager


class ProgressTrackingAgent(EventLogMixin, DefaultAgent):
    """Agent that reports per-step progress via :class:`RunBatchProgressManager` and writes an event log."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        config_class: type = EventLogConfig,
        **kwargs,
    ):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> list[dict]:
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()
