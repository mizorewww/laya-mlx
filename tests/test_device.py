import json
import subprocess
import sys
from unittest.mock import patch

from laya_mlx.device import (
    HardwareProfile,
    get_chip_name,
    get_total_memory_gb,
    is_apple_silicon,
    profile_hardware,
    suggest_device_config,
)


def test_get_total_memory_gb_returns_positive():
    mem = get_total_memory_gb()
    assert isinstance(mem, float)
    assert mem > 0.0


def test_get_chip_name_returns_string():
    name = get_chip_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_is_apple_silicon_returns_bool():
    val = is_apple_silicon()
    assert isinstance(val, bool)


def test_profile_hardware_structure():
    profile = profile_hardware()
    assert isinstance(profile, HardwareProfile)
    assert isinstance(profile.platform, str)
    assert isinstance(profile.is_apple_silicon, bool)
    assert profile.total_memory_gb > 0
    assert profile.cpu_cores >= 1
    assert profile.recommended_dtype == "float16"
    assert profile.recommended_quantize in (None, 4, 8)
    assert profile.recommended_batch_size in (4, 8, 16)
    assert isinstance(profile.low_memory_mode, bool)


def test_memory_tier_recommendations_mocked():
    # Tier 1: Low-end 8GB Apple Silicon (M1/M2/M3 base model)
    with patch("laya_mlx.device.get_total_memory_gb", return_value=8.0):
        p8 = profile_hardware()
        assert p8.recommended_quantize == 4
        assert p8.recommended_batch_size == 4
        assert p8.low_memory_mode is True

    # Tier 2: Mid-tier 16GB
    with patch("laya_mlx.device.get_total_memory_gb", return_value=16.0):
        p16 = profile_hardware()
        assert p16.recommended_quantize == 8
        assert p16.recommended_batch_size == 8
        assert p16.low_memory_mode is False

    # Tier 3: High-end 32GB+ (Pro/Max/Ultra)
    with patch("laya_mlx.device.get_total_memory_gb", return_value=32.0):
        p32 = profile_hardware()
        assert p32.recommended_quantize is None
        assert p32.recommended_batch_size == 16
        assert p32.low_memory_mode is False


def test_suggest_device_config():
    cfg = suggest_device_config()
    assert "dtype" in cfg
    assert "batch_size" in cfg
    assert "quantize" in cfg
    assert "low_memory" in cfg
    assert cfg["dtype"] == "float16"


def test_cli_device_subcommand():
    # Text mode
    res = subprocess.run(
        [sys.executable, "-m", "laya_mlx.cli", "device"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Laya-MLX Hardware Profile" in res.stdout
    assert "Platform:" in res.stdout

    # JSON mode
    res_json = subprocess.run(
        [sys.executable, "-m", "laya_mlx.cli", "device", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(res_json.stdout)
    assert "platform" in data
    assert "total_memory_gb" in data
    assert "suggested_config" in data
