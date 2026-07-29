"""
Shared health-payload helper.

Every service's GET /health returns the same standardized envelope so the
supervisor (Phase D2), CI smoke tests, and the dashboard can rely on a single
contract: {service, status, version, ...extra}.

Usage:
    from health import health_payload
    @app.get("/health")
    def health():
        return health_payload("yolo", port=5002, model_loaded=model is not None)
"""
import os

SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "0.1.0")


def health_payload(service: str, status: str = "ok", version: str | None = None, **extra):
    """Return the standardized health envelope.

    service : service name (required, non-empty)
    status  : "ok" | "degraded" | "down"
    version : overrides SERVICE_VERSION env if given
    extra   : service-specific fields (port, model flags, etc.)
    """
    if not service:
        raise ValueError("health_payload: service name is required")
    payload = {
        "service": service,
        "status": status,
        "version": version or SERVICE_VERSION,
    }
    payload.update(extra)
    return payload
