"""Site customization for ELSA_INT runs.

Intercepts Python startup before /raid/jess/SICNet/timm/sitecustomize.py can run.
That file creates a fake torchvision stub (no get_image_size), which breaks training.
This file imports the real torchvision first, preventing the stub from activating.
"""
from __future__ import annotations
import importlib.abc, sys


class _SicTritonBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if "sic_triton" in fullname:
            raise RuntimeError("sic_triton is blocked by ELSA_INT/sitecustomize.py")
        return None

if not any(isinstance(h, _SicTritonBlocker) for h in sys.meta_path):
    sys.meta_path.insert(0, _SicTritonBlocker())

try:
    import torchvision  # noqa: F401  — must come before /raid/jess/ stub activates
except ImportError:
    pass
