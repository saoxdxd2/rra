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
        self._dispatch_cache = {}
        self._cached_loader = None

    @property
    def loader(self):
        if self._loader is not None:
            return self._loader
        if self._cached_loader is None:
            self._cached_loader = get_cpp_loader()
        return self._cached_loader

    def _get_op(self, op_name, dev_type):
        cache_key = (op_name, dev_type)
        if cache_key in self._dispatch_cache:
            return self._dispatch_cache[cache_key]
        
        mod = self.loader
        if mod is None:
            return None
            
        op = None
        # Try specialized first: op_name_cuda, op_name_cpu
        special_name = f"{op_name}_{dev_type}"
        if hasattr(mod, special_name):
            op = getattr(mod, special_name)
        elif hasattr(mod, op_name):
            # Check if generic op supports the device
            if dev_type == "cpu":
                op = getattr(mod, op_name)
            elif hasattr(mod, "is_cuda_optimized") and mod.is_cuda_optimized():
                op = getattr(mod, op_name)
        
        self._dispatch_cache[cache_key] = op
        return op

    def ready(self, op_name=None, *tensors):
        # Backward compatibility: allow ready(*tensors) call style.
        if op_name is not None and not isinstance(op_name, str):
            tensors = (op_name,) + tensors
            op_name = None

        mod = self.loader
        if mod is None:
            return False
        
        if not tensors:
            return True

        # Determine dominant device
        dev_type = "cpu"
        for t in tensors:
            if isinstance(t, torch.Tensor):
                dev_type = t.device.type
                break
        
        # If op_name provided, check for device-specific implementation
        if op_name is not None:
            # Look for op_name_cuda or op_name_cpu if specialized
            specialized_name = f"{op_name}_{dev_type}"
            if hasattr(mod, specialized_name):
                return True
            # Fallback to base name check
            if not hasattr(mod, op_name):
                return False

        # If we reach here, we are checking if the generic loader can handle the device
        # Currently, the core kernels in cpp_loader are CPU-only
        if dev_type != "cpu":
            # Check if the loader explicitly supports cuda (future)
            if hasattr(mod, "is_cuda_optimized") and mod.is_cuda_optimized():
                return True
            return False
            
        return True

    def has(self, op_name, *tensors):
        return self.ready(op_name, *tensors)

    def missing_ops(self, op_names):
        mod = self.loader
        if mod is None:
            return list(op_names)
        return [op for op in op_names if not hasattr(mod, op)]

    def call(self, op_name, *args, tensors=None):
        # Hot-path optimization: bypass overhead for common calls
        probe_tensors = args if tensors is None else tensors
        dev_type = "cpu"
        # Fast device detection
        for t in probe_tensors:
            if hasattr(t, "device"):
                dev_type = t.device.type
                break
        
        op = self._get_op(op_name, dev_type)
        if op is not None:
            try:
                return op(*args)
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(f"C++ op '{op_name}' failed on '{dev_type}': {exc}") from exc
                return None

        if self.strict:
            if self.loader is None:
                raise RuntimeError(f"C++ loader not available; required op '{op_name}'.")
            raise RuntimeError(f"C++ op '{op_name}' is unavailable or unsupported on device '{dev_type}'.")
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
    return get_accelerator(loader=loader).ready(None, *tensors)


def cpp_has(op_name, *tensors, loader=None):
    return get_accelerator(loader=loader).has(op_name, *tensors)


def accelerate(op_name, *args, tensors=None, loader=None):
    return get_accelerator(loader=loader).call(op_name, *args, tensors=tensors)
