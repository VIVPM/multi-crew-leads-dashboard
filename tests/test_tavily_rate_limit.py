"""No-network checks for the shared per-key Tavily request budget and bounded retry."""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ.setdefault("LLM_MODEL", "GEMINI")

import pipeline  # noqa: E402
from tavily import UsageLimitExceededError  # noqa: E402


class SearchClient:
    """Record searches instead of contacting Tavily."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
        self.reject = None

    def search(self, query, max_results):
        assert max_results == 5
        with self.lock:
            self.calls.append((query, time.monotonic()))
            if self.reject:
                error, self.reject = self.reject, None
                raise error
        return {"results": [query]}


client = SearchClient()
with patch.object(pipeline, "TavilyClient", return_value=client), \
     patch.object(pipeline, "TAVILY_RPM_LIMIT", 2), \
     patch.object(pipeline, "TAVILY_WINDOW_S", 0.08):
    pipeline._TAVILY_CALLS.clear()
    pipeline._TAVILY_BLOCKED_UNTIL.clear()
    tool = pipeline.make_tavily_tool("first-key")
    tool.invoke("one")
    tool.invoke("one")
    assert len(client.calls) == 1, "repeated search should come from the job cache"
    tool.invoke("two")
    before = time.monotonic()
    tool.invoke("three")
    assert time.monotonic() - before >= 0.06, "third request did not wait for capacity"
    before = time.monotonic()
    pipeline.make_tavily_tool("other-key").invoke("four")
    assert time.monotonic() - before < 0.06, "unrelated API keys must not share a limit"

client = SearchClient()
with patch.object(pipeline, "TavilyClient", return_value=client), \
     patch.object(pipeline, "TAVILY_RPM_LIMIT", 3), \
     patch.object(pipeline, "TAVILY_WINDOW_S", 0.1):
    pipeline._TAVILY_CALLS.clear()
    pipeline._TAVILY_BLOCKED_UNTIL.clear()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: pipeline.make_tavily_tool("shared-key").invoke(f"q{i}"), range(8)))
    times = sorted(at for _, at in client.calls)
    assert len(times) == 8
    for i, at in enumerate(times):
        assert sum(at - t < 0.1 for t in times[:i + 1]) <= 3

client = SearchClient()
client.reject = UsageLimitExceededError("Your request has been blocked due to excessive requests.")
with patch.object(pipeline, "TavilyClient", return_value=client), \
     patch.object(pipeline, "TAVILY_RETRY_S", 0.05):
    pipeline._TAVILY_CALLS.clear()
    pipeline._TAVILY_BLOCKED_UNTIL.clear()
    before = time.monotonic()
    pipeline.make_tavily_tool("retry-key").invoke("search")
    assert len(client.calls) == 2 and time.monotonic() - before >= 0.05

client = SearchClient()
client.reject = UsageLimitExceededError("This request exceeds the pay-as-you-go limit.")
with patch.object(pipeline, "TavilyClient", return_value=client):
    pipeline._TAVILY_CALLS.clear()
    pipeline._TAVILY_BLOCKED_UNTIL.clear()
    try:
        pipeline.make_tavily_tool("quota-key").invoke("search")
        raise AssertionError("credit exhaustion must propagate immediately")
    except UsageLimitExceededError:
        assert len(client.calls) == 1


class AlwaysRateLimited:
    def __init__(self):
        self.calls = 0

    def search(self, query, max_results):
        self.calls += 1
        raise UsageLimitExceededError("Your request has been blocked due to excessive requests.")


client = AlwaysRateLimited()
with patch.object(pipeline, "TavilyClient", return_value=client), \
     patch.object(pipeline, "TAVILY_RETRY_S", 0.01):
    pipeline._TAVILY_CALLS.clear()
    pipeline._TAVILY_BLOCKED_UNTIL.clear()
    try:
        pipeline.make_tavily_tool("still-limited-key").invoke("search")
        raise AssertionError("Tavily rejection must stop after one retry")
    except UsageLimitExceededError:
        assert client.calls == 2

print("test_tavily_rate_limit.py: all checks passed")
