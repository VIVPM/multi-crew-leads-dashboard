"""Pure queue scheduling helpers.

The jobs table is the durable queue, so fairness is decided immediately before
the worker's existing conditional claim.  Keeping the selection policy here
makes it testable without importing the worker (which initializes providers,
Supabase, and telemetry at import time).
"""

from typing import Optional


def choose_round_robin_job(
    pending_jobs: list[dict], last_user_id: Optional[str]
) -> Optional[dict]:
    """Choose the oldest job for the tenant after ``last_user_id``.

    ``pending_jobs`` must be ordered oldest first.  The first row seen for each
    user is therefore that user's oldest job.  Cycling through those users
    prevents one tenant's burst from monopolizing a single worker while
    preserving FIFO order within each tenant.
    """
    if not pending_jobs:
        return None

    oldest_by_user: dict[str, dict] = {}
    tenant_order: list[str] = []
    for job in pending_jobs:
        user_id = str(job.get("user_id") or "")
        if user_id not in oldest_by_user:
            oldest_by_user[user_id] = job
            tenant_order.append(user_id)

    if last_user_id in oldest_by_user:
        next_index = (tenant_order.index(last_user_id) + 1) % len(tenant_order)
    else:
        next_index = 0
    return oldest_by_user[tenant_order[next_index]]
