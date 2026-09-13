# src/service/jobs.py
#
# A tiny in-memory job registry so the Streamlit UI can kick off a slow operation
# (cloning + indexing a repo, running the agent, running a whole dataset arm) in a
# background thread and poll it instead of blocking the page.
#
# Deliberately not tied to Streamlit — st.session_state is per-browser-tab and gets
# rebuilt on every rerun, so the job itself has to live in plain module-level state
# that survives reruns. This module is the only place that state lives.
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from src.service.errors import JobCancelled

_JOBS: dict[str, "Job"] = {}
_REGISTRY_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    kind: str                              # "localize" | "evaluate"
    status: str = "pending"                # pending | running | cancelling | done | error | cancelled
    cancelled: bool = False
    trace: list = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def request_cancel(self) -> None:
        """Called from the UI thread when the user clicks Cancel.

        Sets both the flag the worker thread polls (`cancelled`) and a status the
        UI can render immediately — before the worker has had any chance to notice
        the flag — so a single click visibly registers instead of inviting repeat
        clicks.
        """
        self.cancelled = True
        if self.status == "running":
            self.status = "cancelling"


def create_job(kind: str) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], kind=kind)
    with _REGISTRY_LOCK:
        _JOBS[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def any_job_running() -> bool:
    """Backs the UI's "one job at a time" guard.

    src/llm.py's provider/config are process-global (configure()/set_provider()
    mutate module state), so two concurrent jobs with different provider overrides
    would race. Rather than thread that state through every call site, the UI simply
    refuses to start a second job while one is running.
    """
    return any(j.status in ("pending", "running") for j in _JOBS.values())


def start_thread(job: Job, target: Callable, *args, **kwargs) -> None:
    """Run `target(job, *args, **kwargs)` in a daemon thread, updating job.status."""

    def runner():
        if job.status != "cancelling":
            job.status = "running"
        try:
            target(job, *args, **kwargs)
            job.status = "done"
        except JobCancelled:
            job.status = "cancelled"
        except Exception as e:
            job.error = str(e)
            job.status = "error"
        finally:
            job.finished_at = time.time()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
