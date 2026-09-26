"""Shared WebGPU device, storage buffers, uniform blocks and compute kernels."""

from __future__ import annotations

import functools
import struct
from collections.abc import Sequence
from importlib import resources

import numpy as np

# Limits raised to what the adapter offers; WebGPU's defaults (8 storage buffers, 128 MiB) are too small.
_LIMITS = (
    "max-storage-buffer-binding-size",
    "max-buffer-size",
    "max-storage-buffers-per-shader-stage",
    "max-compute-workgroups-per-dimension",
)
MAX_GROUPS = 65535


class GpuUnavailable(RuntimeError):
    """No WebGPU adapter on this machine."""


@functools.cache
def _adapter():
    try:
        import wgpu
    except ImportError as exc:
        raise GpuUnavailable(f"wgpu is not installed: {exc}") from exc
    from wgpu.backends.wgpu_native.extras import set_instance_extras

    # Probing wgpu's GL backend makes its own GL context current under vispy's.
    set_instance_extras(backends=["Vulkan", "Metal", "DX12"])
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise GpuUnavailable("no WebGPU adapter")
    return adapter


@functools.cache
def device():
    """The process-wide wgpu device; raises GpuUnavailable without a GPU."""
    adapter = _adapter()
    limits = adapter.limits
    return adapter.request_device_sync(label="scan-kit", required_limits={k: limits[k] for k in _LIMITS})


def adapter_info() -> dict:
    """Vendor, device, backend and adapter type, for provenance and test gating."""
    info = _adapter().info
    return {k: info.get(k, "") for k in ("vendor", "device", "description", "adapter_type", "backend_type")}


def wgsl_asset(name: str) -> str:
    return (resources.files("scan_kit") / "assets" / name).read_text(encoding="utf-8")


def storage(data=None, *, size: int | None = None, label: str = ""):
    """Read-write storage buffer from *data* (any array) or of *size* zeroed bytes."""
    import wgpu

    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    if data is not None:
        raw = np.ascontiguousarray(data)
        if raw.nbytes == 0:
            raw = np.zeros(1, dtype=np.uint32)
        return device().create_buffer_with_data(data=raw.tobytes(), usage=usage, label=label)
    return device().create_buffer(size=max(int(size or 0), 4), usage=usage, label=label)


def uniform(nbytes: int, label: str = ""):
    import wgpu

    size = (max(int(nbytes), 16) + 15) // 16 * 16
    return device().create_buffer(size=size, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST, label=label)


def read(buffer, dtype, count: int | None = None, offset: int = 0) -> np.ndarray:
    size = None if count is None else int(count) * np.dtype(dtype).itemsize
    return np.frombuffer(device().queue.read_buffer(buffer, offset, size), dtype=dtype).copy()


@functools.cache
def _fence():
    return storage(size=4, label="fence")


def wait() -> None:
    """Block until submitted work finishes (a read waits on the queue before it)."""
    # wgpu-py 0.32's on_submitted_work_done_sync has a broken callback signature.
    device().queue.read_buffer(_fence(), 0, 4)


def workgroups_1d(n: int, size: int = 256) -> tuple[int, int]:
    """(x, y) workgroup counts covering *n* items; kernels index ``gid.x + gid.y * nwg.x * size``."""
    groups = max((int(n) + size - 1) // size, 1)
    x = min(groups, MAX_GROUPS)
    return x, (groups + x - 1) // x


class Params:
    """A uniform block of 4-byte scalars, declared once for both WGSL and the host.

    ``Params("P", [("nx", "i"), ("scale", "f"), ...])`` gives :attr:`wgsl` (the struct text)
    and :meth:`pack` (the matching bytes).
    """

    _WGSL = {"f": "f32", "i": "i32", "u": "u32"}

    def __init__(self, name: str, fields: Sequence[tuple[str, str]]) -> None:
        self.name = name
        self.fields = list(fields)
        body = "".join(f"    {n}: {self._WGSL[t]},\n" for n, t in self.fields)
        self.wgsl = f"struct {name} {{\n{body}}}\n"
        self.nbytes = (4 * len(self.fields) + 15) // 16 * 16

    def pack(self, **values) -> bytes:
        missing = {n for n, _ in self.fields} - values.keys()
        if missing:
            raise KeyError(f"{self.name} is missing {sorted(missing)}")
        raw = b"".join(struct.pack("<" + t.replace("u", "I"), values[n]) for n, t in self.fields)
        return raw.ljust(self.nbytes, b"\0")


class Kernel:
    """One WGSL compute entry point with an automatic bind-group layout."""

    def __init__(self, source: str, *, entry: str = "main", label: str = "") -> None:
        dev = device()
        module = dev.create_shader_module(code=source, label=label)
        self.pipeline = dev.create_compute_pipeline(
            layout="auto", compute={"module": module, "entry_point": entry}, label=label,
        )
        self._layout = self.pipeline.get_bind_group_layout(0)

    def bind(self, *buffers):
        """Bind group with ``buffers[i]`` at ``@binding(i)``."""
        entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(buffers)]
        return device().create_bind_group(layout=self._layout, entries=entries)

    def run(self, group, x: int, y: int = 1, z: int = 1) -> None:
        dev = device()
        encoder = dev.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self.pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups(int(x), int(y), int(z))
        compute.end()
        dev.queue.submit([encoder.finish()])
