"""
Model registry resolver — replaces a hardcoded local weights path with a DB
lookup against `model_versions` (status='active'). Closes the loop on the
human-feedback retraining pipeline (Phase 9): promoting a retrained model is
now `PATCH /api/models/:id/activate`, not a manual file copy + server restart.

Falls back to the static MODELS_DIR/<filename> path passed in if the registry
has no active row for that model_name yet (e.g. fresh install before anyone
has registered a version) — so this never blocks startup.
"""
import os
import logging
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "models", "_registry_cache")


def _fetch_active_version(model_name: str):
    """Returns {version, weights_url} for the active row, or None."""
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared"))
        from db_client import get_conn
    except Exception as exc:
        logger.warning("model_manager: DB client unavailable (%s) — skipping registry lookup", exc)
        return None

    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT version, weights_url FROM model_versions
                WHERE model_name = %s AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
                """,
                (model_name,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as exc:
        logger.warning("model_manager: registry lookup failed for %s (%s) — falling back to static path", model_name, exc)
        return None


def _download_to_cache(url: str, version: str, model_name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    dest = os.path.join(CACHE_DIR, f"{model_name}__{version}.pt")
    if os.path.exists(dest):
        return dest

    logger.info("model_manager: downloading %s version %s from registry -> %s", model_name, version, dest)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(resp.content)
    os.replace(tmp, dest)
    return dest


def resolve_weights_path(model_name: str, static_fallback_path: str) -> str:
    """
    Returns a local filesystem path to load with ultralytics YOLO(). Tries the
    registry's active version first; falls back to static_fallback_path
    (the original models/best.pt-style env-configured path) on any failure.
    """
    active = _fetch_active_version(model_name)
    if not active:
        logger.info("model_manager: no active registry version for %s — using static path %s", model_name, static_fallback_path)
        return static_fallback_path

    try:
        path = _download_to_cache(active["weights_url"], active["version"], model_name)
        logger.info("model_manager: resolved %s -> registry version %s (%s)", model_name, active["version"], path)
        return path
    except Exception as exc:
        logger.warning("model_manager: failed to fetch registry version %s for %s (%s) — falling back to static path", active["version"], model_name, exc)
        return static_fallback_path
