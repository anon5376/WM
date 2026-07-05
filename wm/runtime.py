from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    selected: str
    mps_available: bool
    reason: str


def select_device(requested: str = "auto") -> DeviceInfo:
    import torch

    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "mps" and has_mps:
        return DeviceInfo(requested, "mps", has_mps, "requested mps")
    if requested == "mps":
        return DeviceInfo(requested, "cpu", has_mps, "mps requested but unavailable")
    if requested == "cpu":
        return DeviceInfo(requested, "cpu", has_mps, "requested cpu")
    if has_mps:
        return DeviceInfo(requested, "mps", has_mps, "auto selected mps")
    return DeviceInfo(requested, "cpu", has_mps, "auto fallback to cpu")


def smoke_report() -> dict[str, object]:
    import torch

    info = select_device("auto")
    return {
        "torch_version": torch.__version__,
        "requested_device": info.requested,
        "selected_device": info.selected,
        "mps_available": info.mps_available,
        "device_reason": info.reason,
    }

