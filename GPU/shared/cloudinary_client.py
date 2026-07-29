"""Shared Cloudinary upload helpers used by all Python services.

Wrapped in a circuit breaker (D3): when Cloudinary is failing, the breaker opens
and upload_frame() fails fast (CircuitOpen) instead of every frame upload hanging
on a 30s timeout — preventing a slow dependency from stalling frame extraction.
Callers' per-frame try/except skips the frame; the breaker probes for recovery.
"""
import os
import cloudinary
import cloudinary.uploader

from circuit_breaker import CircuitBreaker

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

_CLOUDINARY_BREAKER = CircuitBreaker(
    "cloudinary",
    failure_threshold=int(os.environ.get("CLOUDINARY_CB_THRESHOLD", "5")),
    reset_timeout_s=float(os.environ.get("CLOUDINARY_CB_RESET", "30")),
)


def _do_upload(file_path: str, folder: str) -> dict:
    result = cloudinary.uploader.upload(
        file_path,
        folder=folder,
        resource_type="image",
        format="jpg",
    )
    return {"url": result["secure_url"], "public_id": result["public_id"]}


def upload_frame(file_path: str, folder: str) -> dict:
    return _CLOUDINARY_BREAKER.call(_do_upload, file_path, folder)


def upload(*args, **kwargs) -> dict:
    """Generic breaker-wrapped passthrough to cloudinary.uploader.upload — for
    callers that need custom params (bytes payload, public_id, eager, etc.)."""
    return _CLOUDINARY_BREAKER.call(cloudinary.uploader.upload, *args, **kwargs)
