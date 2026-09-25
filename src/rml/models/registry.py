"""Name -> model-builder registry. Importing this module does not import PyTorch.

Add an architecture by decorating its class with ``@register_model("name")``
in a module listed in ``_BUILTIN_MODULES``; experiments then select it with
``model.name`` in their config.
"""

from __future__ import annotations

import importlib
from typing import Callable

_REGISTRY: dict[str, Callable] = {}
_BUILTIN_MODULES = ("rml.models.cldnn",)
_loaded = False


def register_model(name: str) -> Callable[[Callable], Callable]:
    def decorator(builder: Callable) -> Callable:
        if name in _REGISTRY and _REGISTRY[name] is not builder:
            raise ValueError(f"Model {name!r} is already registered")
        _REGISTRY[name] = builder
        return builder

    return decorator


def _load_builtins() -> None:
    global _loaded
    if not _loaded:
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)
        _loaded = True


def available_models() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)


def build_model(name: str, *, in_channels: int, num_classes: int, **params):
    _load_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](in_channels=in_channels, num_classes=num_classes, **params)
