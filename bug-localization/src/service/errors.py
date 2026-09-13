# src/service/errors.py
#
# Shared between the LangGraph agent loop (src/localizer/agent.py), the offline
# experiment driver (src/localizer/run.py) and the background-job runner
# (src/service/jobs.py) — kept dependency-free so none of them need to import
# each other just for this one type.


class JobCancelled(Exception):
    """Raised when a job's cancel flag is observed mid-run.

    Deliberately NOT caught by the generic `except Exception` blocks those loops
    already use for ordinary sample/instance failures — every site that can raise
    this must catch it separately and re-raise, so cancellation actually stops the
    job instead of being logged as a failure and continuing to the next iteration.
    """
