"""Shared backend helpers for optional QCANT accelerator execution."""

from __future__ import annotations

from typing import Any, Mapping, Optional
import warnings


_ARRAY_BACKEND_CHOICES = {"auto", "numpy", "cupy"}

__all__ = [
    "build_qml_device",
    "import_cupy",
    "is_gpu_device_name",
    "normalize_array_backend",
    "resolve_array_module",
    "resolve_gpu_parallelism",
    "to_host_array",
    "to_plain_data",
]


def normalize_array_backend(array_backend: Optional[str]) -> str:
    """Normalize user-facing dense array backend selection."""
    normalized = str(array_backend or "auto").strip().lower()
    if normalized not in _ARRAY_BACKEND_CHOICES:
        raise ValueError("array_backend must be one of {'auto', 'numpy', 'cupy'}")
    return normalized


def is_gpu_device_name(device_name: Optional[str]) -> bool:
    """Return True when the requested PennyLane device name targets a GPU path."""
    if device_name is None:
        return False
    normalized = str(device_name).strip().lower()
    return normalized.endswith(".gpu") or "gpu" in normalized or "cuda" in normalized


def import_cupy():
    """Import CuPy or raise a guidance-rich ImportError."""
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CuPy is required for array_backend='cupy'. Install QCANT with GPU extras "
            "or add `cupy-cuda12x` to the current environment."
        ) from exc
    return cp


def resolve_array_module(
    *,
    array_backend: Optional[str],
    device_name: Optional[str] = None,
    allow_gpu: bool = True,
    gpu_fallback_reason: Optional[str] = None,
    context: str,
):
    """Resolve the numerical array module for dense linear algebra.

    ``array_backend="auto"`` preserves CPU behavior unless a GPU PennyLane
    device name clearly indicates that the caller requested a GPU-backed path.
    Explicit ``array_backend="cupy"`` raises if CuPy cannot be imported.
    """
    import numpy as np

    normalized = normalize_array_backend(array_backend)
    gpu_requested = normalized == "cupy" or (normalized == "auto" and is_gpu_device_name(device_name))

    if gpu_requested and not allow_gpu:
        message = (
            f"{context} only supports CPU execution for this code path. "
            f"{gpu_fallback_reason or 'Falling back to NumPy.'}"
        )
        if normalized == "cupy":
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning)
        gpu_requested = False

    if gpu_requested:
        try:
            cp = import_cupy()
        except ImportError:
            if normalized == "cupy":
                raise
            warnings.warn(
                f"{context} requested a GPU path via device_name={device_name!r}, but CuPy is not installed. "
                "Falling back to NumPy.",
                RuntimeWarning,
            )
        else:
            return cp, "cupy", True

    return np, "numpy", False


def resolve_gpu_parallelism(
    *,
    device_name: Optional[str],
    worker_count: int,
    parallel_backend: str,
    context: str,
) -> tuple[int, str]:
    """Clamp worker-pool settings that are unsafe for single-GPU execution."""
    resolved_workers = int(worker_count)
    resolved_backend = str(parallel_backend)

    if not is_gpu_device_name(device_name):
        return resolved_workers, resolved_backend

    if resolved_backend == "process":
        warnings.warn(
            f"{context} does not use process pools on GPU-backed devices. "
            "Downgrading parallel_backend='process' to 'thread'.",
            RuntimeWarning,
        )
        resolved_backend = "thread"

    if resolved_workers > 1:
        warnings.warn(
            f"{context} defaults to one worker on GPU-backed devices to avoid oversubscribing a single GPU. "
            f"Clamping max_workers from {resolved_workers} to 1.",
            RuntimeWarning,
        )
        resolved_workers = 1

    return resolved_workers, resolved_backend


def build_qml_device(
    qml,
    *,
    device_name: Optional[str],
    wires: int,
    device_kwargs: Optional[Mapping[str, Any]] = None,
    shots: Optional[int] = None,
    default_name: str = "lightning.qubit",
    force_default: bool = False,
):
    """Create a PennyLane device with common QCANT defaults.

    Explicit device requests are attempted first. If the requested device is not
    available, QCANT warns and falls back to ``default.qubit`` so CPU smoke tests
    and non-GPU environments remain usable.
    """
    kwargs = dict(device_kwargs or {})
    kwargs.pop("wires", None)
    kwargs.pop("shots", None)
    if shots is not None and shots > 0:
        kwargs["shots"] = shots

    if force_default:
        return qml.device("default.qubit", wires=wires, **kwargs)

    if device_name is not None:
        try:
            return qml.device(device_name, wires=wires, **kwargs)
        except Exception as exc:
            warnings.warn(
                f"Could not construct PennyLane device {device_name!r}; falling back to 'default.qubit'. "
                f"Original error: {exc}",
                RuntimeWarning,
            )
            return qml.device("default.qubit", wires=wires, **kwargs)

    try:
        return qml.device(default_name, wires=wires, **kwargs)
    except Exception as exc:
        warnings.warn(
            f"Could not construct default PennyLane device {default_name!r}; falling back to 'default.qubit'. "
            f"Original error: {exc}",
            RuntimeWarning,
        )
        return qml.device("default.qubit", wires=wires, **kwargs)


def to_host_array(value, *, dtype=None):
    """Convert NumPy/CuPy arrays into a NumPy host array."""
    import numpy as np

    try:
        import cupy as cp
    except ImportError:  # pragma: no cover
        cp = None

    if cp is not None and isinstance(value, cp.ndarray):
        arr = cp.asnumpy(value)
        return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)
    if hasattr(value, "get") and cp is not None:  # cupy scalar-like
        arr = cp.asnumpy(cp.asarray(value))
        return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)
    return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)


def to_plain_data(value):
    """Convert nested NumPy/CuPy containers into JSON-serializable Python data."""
    import numpy as np

    try:
        import cupy as cp
    except ImportError:  # pragma: no cover
        cp = None

    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if cp is not None and isinstance(value, cp.ndarray):
        return cp.asnumpy(value).tolist()
    if cp is not None and isinstance(value, cp.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
