import importlib
import os
from functools import lru_cache

import torch


def _configure_windows_dll_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates = [os.getcwd()]
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    candidates.append(torch_lib)
    seen = set()
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen or not os.path.isdir(path):
            continue
        seen.add(norm)
        try:
            os.add_dll_directory(path)
        except OSError:
            continue


@lru_cache(maxsize=1)
def _load_cpp_loader():
    _configure_windows_dll_dirs()
    for module_name in ("cpp_loader", "cpp_loader_optimized"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def get_cpp_loader():
    return _load_cpp_loader()


class Accelerator:
    """Unified dispatcher for strict C++ kernel execution."""

    def __init__(self, loader=None, strict=True):
        self._loader = loader
        self.strict = bool(strict)

    @property
    def loader(self):
        return get_cpp_loader() if self._loader is None else self._loader

    def ready(self, *tensors):
        mod = self.loader
        if mod is None:
            return False
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.device.type != "cpu":
                return False
        return True

    def has(self, op_name, *tensors):
        mod = self.loader
        if mod is None or not hasattr(mod, op_name):
            return False
        return self.ready(*tensors)

    def missing_ops(self, op_names):
        mod = self.loader
        if mod is None:
            return list(op_names)
        return [op for op in op_names if not hasattr(mod, op)]

    def call(self, op_name, *args, tensors=None):
        mod = self.loader
        probe_tensors = args if tensors is None else tensors
        if mod is None:
            if self.strict:
                raise RuntimeError(f"C++ loader not available; required op '{op_name}'.")
            return None
        if not hasattr(mod, op_name):
            if self.strict:
                raise RuntimeError(f"C++ op '{op_name}' is unavailable in loaded extension.")
            return None
        if not self.ready(*probe_tensors):
            if self.strict:
                raise RuntimeError(f"C++ op '{op_name}' requires CPU tensors.")
            return None
        try:
            return getattr(mod, op_name)(*args)
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"C++ op '{op_name}' failed: {exc}") from exc
            return None

    def configure(self, **kwargs):
        mod = self.loader
        if mod is None:
            if self.strict:
                raise RuntimeError("C++ loader not available for configure().")
            return
        for name, value in kwargs.items():
            if hasattr(mod, name):
                getattr(mod, name)(*value if isinstance(value, tuple) else value)


_DEFAULT_ACCEL = Accelerator()


def get_accelerator(loader=None, strict=True):
    if loader is None and strict and _DEFAULT_ACCEL.strict:
        return _DEFAULT_ACCEL
    return Accelerator(loader=loader, strict=strict)


def cpp_ready(*tensors, loader=None):
    return get_accelerator(loader=loader).ready(*tensors)


def cpp_has(op_name, *tensors, loader=None):
    return get_accelerator(loader=loader).has(op_name, *tensors)


def accelerate(op_name, *args, tensors=None, loader=None):
    return get_accelerator(loader=loader).call(op_name, *args, tensors=tensors)
