#!/usr/bin/env python3
"""
BlendColab / GodotBakeCloud — Standalone Godot 4 Vulkan + Xvfb Lightmap Bake Runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from env_bootstrap import inject_godot_addon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("GodotBakeRunner")


def verify_vulkan_environment() -> str:
    """Checks that an NVIDIA or Mesa Vulkan ICD JSON exists for headless Vulkan."""
    icd_candidates = [
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
        Path("/etc/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"),
    ]
    for icd in icd_candidates:
        if icd.exists():
            os.environ.setdefault("VK_ICD_FILENAMES", str(icd))
            return str(icd)
    log.warning("No explicit Vulkan ICD JSON found in standard paths; relying on loader default.")
    return "default"


def bake_scene(
    godot_bin: Path,
    project_dir: Path,
    scene_res_path: str,
    repo_dir: Path | None = None,
    timeout_sec: int = 1800,
) -> dict[str, object]:
    if not shutil.which("xvfb-run"):
        raise RuntimeError("`xvfb-run` is not installed. Run `apt-get install -y xvfb libvulkan1`.")

    icd = verify_vulkan_environment()
    log.info("Using Vulkan ICD: %s", icd)

    if repo_dir is not None:
        inject_godot_addon(repo_dir, project_dir)

    report_path = Path("/tmp") / f"godot_bake_{uuid.uuid4().hex[:8]}.json"
    cmd = [
        "xvfb-run",
        "-a",
        "-s",
        "-screen 0 1280x720x24",
        str(godot_bin),
        "--editor",
        "--rendering-driver",
        "vulkan",
        "--path",
        str(project_dir),
        "--",
        f"--bake-scene={scene_res_path}",
        f"--bake-report={report_path}",
    ]

    log.info("Launching Godot 4 Editor under Xvfb for %s...", scene_res_path)
    proc = subprocess.run(cmd, text=True, timeout=timeout_sec)
    if proc.returncode != 0:
        raise RuntimeError(f"Godot exited with return code {proc.returncode}")

    if not report_path.exists():
        raise RuntimeError("Godot terminated without generating a bake report JSON.")

    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Godot 4 LightmapGI Baker")
    parser.add_argument("--godot-bin", required=True, help="Path to Godot 4 Linux x86_64 binary")
    parser.add_argument("--project", required=True, help="Path to Godot 4 project directory")
    parser.add_argument("--scene", default="res://main.tscn", help="Scene resource path (e.g. res://levels/level1.tscn)")
    parser.add_argument("--repo-dir", default=".", help="Path to BlendColab repo containing godot_addon/")
    args = parser.parse_args()

    result = bake_scene(
        godot_bin=Path(args.godot_bin).resolve(),
        project_dir=Path(args.project).resolve(),
        scene_res_path=args.scene,
        repo_dir=Path(args.repo_dir).resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()