"""Demo-only live view of per-event pipeline progress. Tries a Tkinter
window (stdlib, no extra dependency); falls back to a periodic console table
if Tkinter/display is unavailable (e.g. headless CI) — visualization must
never be a hard dependency of the pipeline itself.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from core.logging_config import get_stage_logger

log = get_stage_logger("visualization")

FIELDS = [
    "event_id", "gap_detected", "ocr_status", "frames_captured",
    "frames_assigned", "frames_stitched", "components_detected",
    "storage_status", "pipeline_stage",
]


@dataclass
class EventViewState:
    event_id: str
    gap_detected: bool = False
    ocr_status: str = "pending"
    frames_captured: int = 0
    frames_assigned: int = 0
    frames_stitched: int = 0
    components_detected: int = 0
    storage_status: str = "pending"
    pipeline_stage: str = "OPEN"


class VisualizationState:
    """Thread-safe shared state — managers update it, the view thread reads it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events: Dict[str, EventViewState] = {}
        self._order = []

    def update(self, event_id: str, **fields) -> None:
        with self._lock:
            if event_id not in self._events:
                self._events[event_id] = EventViewState(event_id=event_id)
                self._order.append(event_id)
            state = self._events[event_id]
            for k, v in fields.items():
                setattr(state, k, v)

    def snapshot(self):
        with self._lock:
            return [self._events[eid] for eid in self._order]


def _console_loop(state: VisualizationState, refresh_ms: int, stop_flag: threading.Event) -> None:
    while not stop_flag.is_set():
        rows = state.snapshot()
        lines = ["=" * 100, "  RAILWAY AI INSPECTION — PHASE-1 PIPELINE MONITOR", "=" * 100]
        header = f"{'EVENT':<12}{'GAP':<6}{'OCR':<12}{'CAP':<6}{'ASGN':<6}{'STCH':<6}{'DET':<6}{'STORE':<10}{'STAGE':<18}"
        lines.append(header)
        for r in rows[-15:]:
            lines.append(
                f"{r.event_id:<12}{('Y' if r.gap_detected else '-'):<6}{r.ocr_status:<12}"
                f"{r.frames_captured:<6}{r.frames_assigned:<6}{r.frames_stitched:<6}"
                f"{r.components_detected:<6}{r.storage_status:<10}{r.pipeline_stage:<18}"
            )
        print("\n".join(lines), flush=True)
        stop_flag.wait(refresh_ms / 1000.0)


def _tk_loop(state: VisualizationState, refresh_ms: int, stop_flag: threading.Event) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Railway AI Inspection — Phase-1 Pipeline Monitor")
    tree = ttk.Treeview(root, columns=FIELDS, show="headings", height=20)
    for col in FIELDS:
        tree.heading(col, text=col.replace("_", " ").upper())
        tree.column(col, width=110, anchor="center")
    tree.pack(fill="both", expand=True)

    def refresh():
        if stop_flag.is_set():
            root.destroy()
            return
        existing = set(tree.get_children())
        for r in state.snapshot():
            values = (
                r.event_id, "YES" if r.gap_detected else "-", r.ocr_status,
                r.frames_captured, r.frames_assigned, r.frames_stitched,
                r.components_detected, r.storage_status, r.pipeline_stage,
            )
            if r.event_id in existing:
                tree.item(r.event_id, values=values)
            else:
                tree.insert("", "end", iid=r.event_id, values=values)
        root.after(refresh_ms, refresh)

    root.after(refresh_ms, refresh)
    root.protocol("WM_DELETE_WINDOW", lambda: stop_flag.set())
    root.mainloop()


class Visualizer:
    def __init__(self, state: VisualizationState, refresh_ms: int = 500):
        self._state = state
        self._refresh_ms = refresh_ms
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        try:
            import tkinter  # noqa: F401
            target = _tk_loop
        except Exception:
            log.warning("Tkinter unavailable — falling back to console visualization")
            target = _console_loop

        self._thread = threading.Thread(
            target=target, args=(self._state, self._refresh_ms, self._stop_flag), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
