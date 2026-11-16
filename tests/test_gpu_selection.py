"""
Tests GPU architecture detection (T4 Turing vs. P100 Pascal), CUDA_VISIBLE_DEVICES
pinning for Kaggle T4 x2 setups, Blender backend routing, and Blender 3.6–5.3 version URLs.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Mock Blender's internal `bpy` module so `blender_gpu_setup` can be tested in standard CI
sys.modules.setdefault("bpy", MagicMock())

import blender_gpu_setup  # noqa: E402
from env_bootstrap import BLENDER_VERSIONS, resolve_blender_url  # noqa: E402
from render_worker import GPUDevice, query_gpus  # noqa: E402


def test_gpu_device_optix_capability_matrix() -> None:
    t4 = GPUDevice(index=0, name="Tesla T4", vram_mb=15360, compute_cap="7.5")
    l4 = GPUDevice(index=0, name="NVIDIA L4", vram_mb=23034, compute_cap="8.9")
    a100 = GPUDevice(index=0, name="NVIDIA A100-SXM4-40GB", vram_mb=40960, compute_cap="8.0")
    p100 = GPUDevice(index=0, name="Tesla P100-PCIE-16GB", vram_mb=16384, compute_cap="6.0")
    v100 = GPUDevice(index=0, name="Tesla V100-SXM2-16GB", vram_mb=16384, compute_cap="7.0")

    assert t4.supports_optix is True
    assert l4.supports_optix is True
    assert a100.supports_optix is True
    assert p100.supports_optix is False
    assert v100.supports_optix is False


def test_query_gpus_filters_cuda_visible_devices(monkeypatch) -> None:
    fake_smi = (
        "0, Tesla T4, 15360, 7.5, 12, 45\n"
        "1, Tesla T4, 15360, 7.5, 88, 63\n"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with patch("subprocess.check_output", return_value=fake_smi):
        gpus = query_gpus()

    assert len(gpus) == 1
    assert gpus[0].index == 1
    assert gpus[0].utilization_pct == 88
    assert gpus[0].temp_c == 63


def test_blender_backend_prefers_optix_on_t4_and_cuda_on_p100() -> None:
    with patch("subprocess.check_output", return_value="7.5, Tesla T4\n"):
        assert blender_gpu_setup.detect_preferred_backend() == "OPTIX"

    with patch("subprocess.check_output", return_value="6.0, Tesla P100-PCIE-16GB\n"):
        assert blender_gpu_setup.detect_preferred_backend() == "CUDA"


def test_all_blender_versions_resolve_cleanly() -> None:
    assert len(BLENDER_VERSIONS) == 11
    for label, expected_url in BLENDER_VERSIONS.items():
        tag, url = resolve_blender_url(label)
        assert url == expected_url
        assert tag in url

    # Also verify short version aliases work
    tag_36, url_36 = resolve_blender_url("3.6")
    assert tag_36 == "3.6.23"
    assert "Blender3.6" in url_36

    tag_53, url_53 = resolve_blender_url("5.3")
    assert tag_53 == "5.3.0"
    assert "Blender5.3" in url_53

    with pytest.raises(ValueError):
        resolve_blender_url("2.79")