"""Probe package.

Importing this module registers every built-in probe, so registry.available() and
registry.get() work off a bare `import agopx.probes`. Without these imports the
registry stays empty and get() raises KeyError for every name -- the decorators only
run when their module is first imported.
"""
from . import offline, online  # noqa: F401  (imported for the @register side effect)
from .base import Probe
from .registry import available, get, register

__all__ = ["Probe", "available", "get", "register", "offline", "online"]
