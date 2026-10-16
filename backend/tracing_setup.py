"""Shared Langfuse tracing setup for the worker and golden evaluation runner."""

import base64
import logging
import os

from logging_setup import current_correlation_ids


def setup_langfuse_tracing(logger: logging.Logger):
    """Instrument LangChain and export spans to Langfuse when both keys are configured."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        logger.info("Langfuse not configured — LLM tracing disabled")
        return None
    try:
        from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.langchain import LangChainInstrumentor

        class _CorrelationSpanProcessor(SpanProcessor):
            """Add the current job and lead IDs to each Langfuse observation."""

            def on_start(self, span, parent_context=None):
                for key, value in current_correlation_ids().items():
                    if value is not None:
                        span.set_attribute(f"langfuse.trace.metadata.{key}", value)
                        span.set_attribute(f"langfuse.observation.metadata.{key}", value)

        provider = TracerProvider(resource=Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", "sales-pipeline-backend"),
            "service.namespace": "lead-coordinator",
            "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
        }))
        provider.add_span_processor(_CorrelationSpanProcessor())
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
        credentials = f'{os.environ["LANGFUSE_PUBLIC_KEY"]}:{os.environ["LANGFUSE_SECRET_KEY"]}'
        auth = base64.b64encode(credentials.encode()).decode()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={
                "Authorization": f"Basic {auth}",
                "x-langfuse-ingestion-version": "4",
            },
        )))
        LangChainInstrumentor().instrument(tracer_provider=provider)
        logger.info("LLM tracing enabled via OTLP: Langfuse (%s)", host)
        return provider
    except Exception:
        logger.exception("Failed to initialize LLM tracing (non-fatal)")
        return None
