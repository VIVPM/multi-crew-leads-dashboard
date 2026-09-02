"""No-network checks for tenant-fair queue scheduling."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from queue_policy import choose_round_robin_job  # noqa: E402


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

print("test_queue_policy.py: all checks passed")
