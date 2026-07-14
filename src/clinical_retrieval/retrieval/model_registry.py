"""Singleton model loaders to avoid repeated GPU/weight init."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_CACHE: dict[str, Any] = {}


def get_or_create(key: str, factory):
    with _lock:
        if key not in _CACHE:
            _CACHE[key] = factory()
        return _CACHE[key]


def clear_registry() -> None:
    with _lock:
        _CACHE.clear()


def has(key: str) -> bool:
    return key in _CACHE
