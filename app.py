from __future__ import annotations

import asyncio.base_events as base_events

from roadsentinel.ui import build_demo, get_launch_kwargs


def _patch_asyncio_event_loop_del() -> None:
    """Suppress known HF Spaces cleanup noise from event-loop finalizers."""
    original_del = getattr(base_events.BaseEventLoop, "__del__", None)
    if original_del is None:
        return
    if getattr(original_del, "__name__", "") == "patched_del":
        return

    def patched_del(self) -> None:
        try:
            original_del(self)
        except ValueError as exc:
            if "Invalid file descriptor" not in str(exc):
                raise

    base_events.BaseEventLoop.__del__ = patched_del


_patch_asyncio_event_loop_del()


demo = build_demo()


if __name__ == "__main__":
    demo.launch(**get_launch_kwargs())
