"""WebGPU compute (Vulkan, DX12 or Metal through wgpu-py): one shared device and WGSL kernels."""

from .core import (
    GpuUnavailable,
    Kernel,
    Params,
    adapter_info,
    device,
    read,
    storage,
    uniform,
    wait,
    wgsl_asset,
    workgroups_1d,
)

__all__ = [
    "GpuUnavailable",
    "Kernel",
    "Params",
    "adapter_info",
    "device",
    "read",
    "storage",
    "uniform",
    "wait",
    "wgsl_asset",
    "workgroups_1d",
]
