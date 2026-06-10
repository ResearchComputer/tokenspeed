"""Backend registry and selection.

Backends self-register with `@register(kernel_name, Backend.X)`. The public op
calls `dispatch(kernel_name, *args, backend="auto", **kwargs)`, which resolves:
explicit arg -> env override (XKERNELS_BACKEND) -> auto (per-vendor preference),
falling back to REFERENCE.
"""
from __future__ import annotations

import os
from collections.abc import Callable

from ._backends import Backend, detect_vendor

# kernel_name -> {Backend: callable}
_REGISTRY: dict[str, dict[Backend, Callable]] = {}

# Per-vendor preference order for "auto" selection (first available wins).
_AUTO_ORDER: dict[str, list[Backend]] = {
    "nvidia": [Backend.CUDA, Backend.TRITON, Backend.REFERENCE],
    "amd": [Backend.HIP, Backend.TRITON, Backend.REFERENCE],
    "none": [Backend.REFERENCE],
}


def register(kernel_name: str, backend: Backend) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        _REGISTRY.setdefault(kernel_name, {})[backend] = fn
        return fn

    return deco


def registered_backends(kernel_name: str) -> list[Backend]:
    return list(_REGISTRY.get(kernel_name, {}).keys())


def _coerce(backend: Backend | str) -> Backend:
    return backend if isinstance(backend, Backend) else Backend(backend)


def select_backend(kernel_name: str, backend: Backend | str = "auto") -> Backend:
    if kernel_name not in _REGISTRY:
        raise KeyError(f"no backends registered for kernel '{kernel_name}'")
    impls = _REGISTRY[kernel_name]

    if backend != "auto":
        chosen = _coerce(backend)
        if chosen not in impls:
            raise KeyError(
                f"backend {chosen.name} not registered for '{kernel_name}'; "
                f"have {[b.name for b in impls]}"
            )
        return chosen

    env = os.environ.get("XKERNELS_BACKEND")
    if env:
        chosen = Backend(env.lower())
        if chosen in impls:
            return chosen

    for candidate in _AUTO_ORDER.get(detect_vendor(), [Backend.REFERENCE]):
        if candidate in impls:
            return candidate
    # Last resort: anything registered.
    return next(iter(impls))


def dispatch(kernel_name: str, *args, backend: Backend | str = "auto", **kwargs):
    chosen = select_backend(kernel_name, backend)
    return _REGISTRY[kernel_name][chosen](*args, **kwargs)
