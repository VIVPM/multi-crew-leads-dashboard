"""Check that Langfuse traces still export through OpenTelemetry without network."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
for name, value in {
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_KEY": "test-key",
    "SECRET_KEY": "test-secret",
    "DAILY_LEAD_CAP": "5",
    "LLM_MODEL": "GEMINI",
    "GEMINI_API_KEY": "test-gemini",
    "TAVILY_API_KEY": "test-tavily",
    "ALLOWED_ORIGINS": "http://localhost:5173",
}.items():
    os.environ.setdefault(name, value)
os.environ["LANGFUSE_PUBLIC_KEY"] = "test-public"
os.environ["LANGFUSE_SECRET_KEY"] = "test-secret"
os.environ["LANGFUSE_HOST"] = "https://example.test"

from opentelemetry.exporter.otlp.proto.http import trace_exporter  # noqa: E402
from opentelemetry.sdk.trace.export import SpanExportResult  # noqa: E402


class FakeExporter:
    """Capture trace exports without sending HTTP requests."""

    instances = []

    def __init__(self, endpoint, headers):
        self.endpoint, self.headers, self.spans = endpoint, headers, []
        self.instances.append(self)

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


trace_exporter.OTLPSpanExporter = FakeExporter
import worker  # noqa: E402

assert len(FakeExporter.instances) == 1, "worker should export only to Langfuse"
exporter = FakeExporter.instances[0]
assert exporter.endpoint == "https://example.test/api/public/otel/v1/traces"
assert exporter.headers["Authorization"].startswith("Basic ")

with worker.job_context("job-42"):
    with worker._tracer_provider.get_tracer("audit-check").start_as_current_span("test-span"):
        pass
assert worker._tracer_provider.force_flush(timeout_millis=5000)
assert any(span.name == "test-span" and span.attributes["langfuse.trace.metadata.job_id"] == "job-42"
           for span in exporter.spans)
print("test_observability.py: all checks passed")
