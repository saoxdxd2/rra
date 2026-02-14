import os
from functools import lru_cache

import torch


def _configure_windows_dll_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    
    # Common candidates: local dir, torch lib, and Intel oneAPI/compiler runtimes
    candidates = [
        os.getcwd(),
        os.path.join(os.path.dirname(torch.__file__), "lib"),
    ]
    
    # Intel Compiler (ICX) runtime paths for OpenMP (libiomp5md.dll)
    intel_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin\intel64",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\redist\intel64_win\compiler",
    ]
    candidates.extend(intel_paths)
    
    # Also check if it's already in the PATH
    env_path = os.environ.get("PATH", "")
    for p in env_path.split(os.pathsep):
        if "intel" in p.lower() or "oneapi" in p.lower():
            candidates.append(p)

    seen = set()
    for path in candidates:
        if not path: continue
        norm = os.path.normpath(path)
        if norm in seen or not os.path.isdir(path):
            continue
        seen.add(norm)
        try:
            os.add_dll_directory(norm)
        except (OSError, ValueError):
            continue


@lru_cache(maxsize=1)
def _load_cpp_loader():
    import sys
    import importlib.util
    import importlib.machinery
    
    root = os.path.dirname(os.path.abspath(__file__))
    # Force project root to front of sys.path
    if root not in sys.path:
        sys.path.insert(0, root)
        
    _configure_windows_dll_dirs()
    
    # Enforce local extension loading only.
    suffixes = list(importlib.machinery.EXTENSION_SUFFIXES)
    for extra in (".cp312-win_amd64.pyd", ".pyd", ".so"):
        if extra not in suffixes:
            suffixes.append(extra)

    attempted_files = []
    load_failures = []

    for name in ("cpp_loader", "cpp_loader_optimized"):
        for suffix in suffixes:
            pyd_path = os.path.join(root, f"{name}{suffix}")
            if not os.path.exists(pyd_path):
                continue
            attempted_files.append(pyd_path)
            try:
                loader = importlib.machinery.ExtensionFileLoader("cpp_loader", pyd_path)
                spec = importlib.util.spec_from_loader("cpp_loader", loader)
                if spec is None or spec.loader is None:
                    raise RuntimeError("importlib returned an invalid extension spec.")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                print(f">>> C++ LOADER ENFORCED LOCAL: {pyd_path}")
                return mod
            except Exception as e:
                load_failures.append(f"{pyd_path}: {type(e).__name__}: {e}")

    if not attempted_files:
        expected = ", ".join([f"cpp_loader{sx}" for sx in suffixes[:6]])
        message = (
            "C++ loader startup failed: no local extension binary found. "
            f"Expected files in '{root}', for example: {expected}."
        )
        print(f">>> C++ LOADER ERROR: {message}")
        raise RuntimeError(message)

    max_errors = 8
    failure_details = "\n".join(load_failures[:max_errors])
    overflow = len(load_failures) - max_errors
    if overflow > 0:
        failure_details = f"{failure_details}\n... ({overflow} more load errors omitted)"

    message = (
        "C++ loader startup failed: local extension binaries were found but all load attempts failed.\n"
        f"Attempted files ({len(attempted_files)}): {attempted_files}\n"
        f"Load errors:\n{failure_details}"
    )
    print(f">>> C++ LOADER ERROR: {message}")
    raise RuntimeError(message)


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
            # Base op-name check when no device-specialized symbol exists.
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
                # STRUCT LATCHING: Check if we are calling the unified dispatcher
                if op_name == "unified_dispatch_io":
                    # Expected args: (x, state, params, scalars, cmd)
                    # New signature expect: ( [x, state, scalars, ...params], cmd )
                    if len(args) == 5:
                        x, state, params, scalars, cmd = args
                        ctx = [x, state, scalars] + list(params)
                        return op(ctx, int(cmd))
                
                return op(*args)
            except Exception as exc:
                raise RuntimeError(
                    f"C++ op '{op_name}' failed on device '{dev_type}'. "
                    f"args={len(args)} strict={self.strict} error={type(exc).__name__}: {exc}"
                ) from exc

        raise RuntimeError(
            f"C++ op '{op_name}' is unavailable or unsupported on device '{dev_type}'. "
            "No Python fallback path is allowed."
        )

    def configure(self, **kwargs):
        mod = self.loader
        for name, value in kwargs.items():
            if hasattr(mod, name):
                getattr(mod, name)(*value if isinstance(value, tuple) else value)
            else:
                raise RuntimeError(f"C++ configure target '{name}' is missing from loader.")


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
