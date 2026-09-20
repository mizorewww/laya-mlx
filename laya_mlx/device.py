"""Hardware detection and device-adaptive configuration for Apple Silicon.

Optimizes execution parameters (precision, batch size, quantization, memory caching)
for low-end devices (e.g. 8GB/16GB unified memory M1/M2/M3) vs high-end hardware.
"""

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    platform: str
    is_apple_silicon: bool
    chip_name: str
    total_memory_gb: float
    cpu_cores: int
    recommended_quantize: Optional[int]
    recommended_batch_size: int
    recommended_dtype: str
    low_memory_mode: bool


def get_total_memory_gb() -> float:
    """Detect total system memory in gigabytes."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return float(out.strip()) / (1024**3)
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        return kb / (1024**2)
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullTotalPhys) / (1024**3)
        except Exception:
            pass

    # Fallback to os.sysconf if available
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024**3)
    except Exception:
        return 16.0  # Default assumption


def get_chip_name() -> str:
    """Return processor/chip description."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
            return out.strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def is_apple_silicon() -> bool:
    """Return True if running on Apple Silicon (arm64 Darwin)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def profile_hardware() -> HardwareProfile:
    """Inspect local hardware and produce an optimal configuration profile.

    Device memory tiers:
    - <= 8 GB RAM (Base M1/M2/M3): Bandwidth & RAM constrained. Recommend Q4 (4-bit),
      small batch size (4), and aggressive cache cleanup.
    - 9-16 GB RAM: Recommend Q8 (8-bit) or FP16 with batch size 8.
    - > 16 GB RAM (Pro/Max/Ultra): Standard FP16 with full batch size.
    """
    total_mem = get_total_memory_gb()
    chip = get_chip_name()
    cores = os.cpu_count() or 4
    apple_silicon = is_apple_silicon()

    if total_mem <= 8.5:
        # Low-end constraint: 8GB Unified Memory
        rec_quantize = 4
        rec_batch = 4
        low_mem = True
    elif total_mem <= 16.5:
        # Mid-tier: 16GB Unified Memory
        rec_quantize = 8
        rec_batch = 8
        low_mem = False
    else:
        # High-end: 24GB, 36GB, 64GB+
        rec_quantize = None
        rec_batch = 16
        low_mem = False

    return HardwareProfile(
        platform=platform.system(),
        is_apple_silicon=apple_silicon,
        chip_name=chip,
        total_memory_gb=round(total_mem, 1),
        cpu_cores=cores,
        recommended_quantize=rec_quantize,
        recommended_batch_size=rec_batch,
        recommended_dtype="float16",
        low_memory_mode=low_mem,
    )


def suggest_device_config() -> dict:
    """Return dictionary of recommended Agent keyword arguments."""
    p = profile_hardware()
    return {
        "dtype": p.recommended_dtype,
        "batch_size": p.recommended_batch_size,
        "quantize": p.recommended_quantize,
        "low_memory": p.low_memory_mode,
    }
