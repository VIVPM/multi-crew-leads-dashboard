"""No-network checks for tenant-fair queue scheduling."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from queue_policy import choose_round_robin_job, next_concurrency  # noqa: E402


JOBS = [
    {"id": "a1", "user_id": "a"},
    {"id": "a2", "user_id": "a"},
    {"id": "b1", "user_id": "b"},
    {"id": "c1", "user_id": "c"},
]

assert choose_round_robin_job([], None) is None
assert choose_round_robin_job(JOBS, None)["id"] == "a1"
assert choose_round_robin_job(JOBS, "a")["id"] == "b1"
assert choose_round_robin_job(JOBS, "b")["id"] == "c1"
assert choose_round_robin_job(JOBS, "c")["id"] == "a1"
assert choose_round_robin_job(JOBS, "departed-tenant")["id"] == "a1"

# --- queue-depth autoscaling of in-worker concurrency ---
# There is no worker count to scale on a single-instance host, so the same
# control loop moves concurrency slots instead.
BOUNDS = dict(minimum=2, maximum=10, scale_up_depth=5, cooldown_s=30)


def step(current, depth, since=999):
    return next_concurrency(current, depth, since, **BOUNDS)


# A deep backlog doubles toward the ceiling and stops there.
assert step(2, 20) == 4
assert step(4, 20) == 8
assert step(8, 20) == 10
assert step(10, 20) == 10, "never exceeds MAX_CONCURRENT_JOBS"

# An empty queue halves back, never below the floor.
assert step(10, 0) == 5
assert step(5, 0) == 2
assert step(2, 0) == 2, "never drops below WORKER_MIN_CONCURRENCY"

# Cooldown is the hysteresis: a queue hovering at the threshold must not
# retarget on every poll.
assert step(10, 0, since=29.9) == 10, "inside cooldown, no scale-in"
assert step(2, 20, since=0) == 4, (
    "scale-up ignores the cooldown — making queued work wait it out would be "
    "worse than the fixed ceiling this replaces")

# Below the scale-up threshold but not empty: hold. Reacting to one queued job
# would flap between levels all day.
assert step(4, 1) == 4
assert step(4, 4) == 4
assert step(4, 5) == 8, "at the threshold, not just above it"

# A floor above the current target still wins, so raising the minimum takes
# effect on the next scale-up rather than being silently clamped away.
assert next_concurrency(1, 9, 999, minimum=4, maximum=10,
                        scale_up_depth=5, cooldown_s=30) == 4

print("test_queue_policy.py: all checks passed")
