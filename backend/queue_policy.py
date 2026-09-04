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


def next_concurrency(
    current: int,
    queue_depth: int,
    seconds_since_change: float,
    *,
    minimum: int,
    maximum: int,
    scale_up_depth: int,
    cooldown_s: float,
) -> int:
    """How many jobs the worker should run at once on the next poll.

    This deployment runs one worker on a single-instance host, so there is no
    worker count to scale.  The equivalent control loop scales the concurrency
    slots inside that worker: queue depth is the signal, ``minimum``/``maximum``
    bound the range, and ``cooldown_s`` supplies the hysteresis that stops a
    queue hovering around the threshold from oscillating every poll.

    Scaling is multiplicative so a backlog is met in a few polls rather than
    one slot at a time, and scale-in only halves, so a queue that empties
    briefly does not immediately give up the capacity it just earned.  The
    caller stops *claiming* when it is over target; running jobs are never
    cancelled, since the model calls have already been paid for.

    The cooldown guards scale-in only.  Making a backlog wait it out would be
    strictly worse than the fixed ceiling this replaces: work is already
    queued, and the cost of another slot is bounded by ``maximum`` either way.
    Releasing capacity is the direction worth being slow and reluctant about.
    """
    if queue_depth >= scale_up_depth and current < maximum:
        return min(maximum, max(current * 2, minimum))  # up immediately
    if queue_depth == 0 and current > minimum and seconds_since_change >= cooldown_s:
        return max(minimum, current // 2)
    return current
