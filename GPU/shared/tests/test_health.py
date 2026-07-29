"""
Unit smoke test for the shared health-payload helper.

This is the ALWAYS-GREEN gate: no torch / paddle / cv2 / network. Just validates
the standardized health envelope contract that every service's /health depends on.
"""
import os
import sys

# Make GPU/shared importable (same path trick the services use).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402
from health import health_payload  # noqa: E402


REQUIRED_KEYS = {"service", "status", "version"}


def test_payload_has_required_keys():
    p = health_payload("yolo")
    assert REQUIRED_KEYS <= set(p.keys())
    assert p["service"] == "yolo"
    assert p["status"] == "ok"
    assert isinstance(p["version"], str) and p["version"]


def test_extra_fields_passthrough():
    p = health_payload("yolo", port=5002, defect_model_loaded=True)
    assert p["port"] == 5002
    assert p["defect_model_loaded"] is True
    # extras must never clobber the required envelope
    assert REQUIRED_KEYS <= set(p.keys())


def test_status_override():
    p = health_payload("ocr", status="degraded")
    assert p["status"] == "degraded"


def test_version_env_override(monkeypatch):
    # version falls back to SERVICE_VERSION env when not passed explicitly
    import importlib
    import health as health_mod
    monkeypatch.setenv("SERVICE_VERSION", "9.9.9")
    importlib.reload(health_mod)
    assert health_mod.health_payload("sync_engine")["version"] == "9.9.9"
    importlib.reload(health_mod)  # restore default for other tests


def test_explicit_version_wins_over_env(monkeypatch):
    monkeypatch.setenv("SERVICE_VERSION", "9.9.9")
    assert health_payload("report_generator", version="1.2.3")["version"] == "1.2.3"


def test_empty_service_rejected():
    with pytest.raises(ValueError):
        health_payload("")


@pytest.mark.parametrize("svc", [
    "yolo", "ocr", "frame_extractor", "sync_engine", "correlation", "report_generator",
])
def test_all_service_names_supported(svc):
    assert health_payload(svc)["service"] == svc
