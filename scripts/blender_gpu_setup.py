"""
Injected Blender startup script (`blender -b scene.blend -P blender_gpu_setup.py -- ...`).

Auto-detects NVIDIA GPU architecture (Turing/Ampere+ vs Pascal P100), selects OPTIX
when RT cores are present with automatic fallback to CUDA, activates OpenImageDenoise,
and applies CLI sample/multi-GPU overrides.
"""

import argparse
import os
import subprocess
import sys
import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--pool-gpus", action="store_true")
    return parser.parse_args(argv)


def detect_preferred_backend() -> str:
    """
    Returns 'OPTIX' for Turing (CC 7.5, e.g., T4) and newer GPUs;
    returns 'CUDA' for Pascal (CC 6.0, e.g., P100) to avoid OptiX kernel stalls.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            cap_str, name = [x.strip() for x in line.split(",", 1)]
            major, minor = map(int, cap_str.split(".")[:2])
            if (major, minor) >= (7, 5):
                print(f"[BlendColab] Detected RT-capable GPU '{name}' (CC {cap_str}) -> Preferring OPTIX")
                return "OPTIX"
            print(f"[BlendColab] Detected non-RT GPU '{name}' (CC {cap_str}) -> Preferring CUDA")
            return "CUDA"
    except Exception as exc:
        print(f"[BlendColab] GPU query fallback ({exc}) -> Trying OPTIX with CUDA fallback")
    return "OPTIX"


def configure_cycles(samples: int | None, pool_gpus: bool) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons["cycles"].preferences

    preferred = detect_preferred_backend()
    backends_to_try = ["OPTIX", "CUDA"] if preferred == "OPTIX" else ["CUDA", "OPTIX"]

    activated_backend = None
    for backend in backends_to_try:
        try:
            cycles_prefs.compute_device_type = backend
            cycles_prefs.get_devices()
            gpu_devices = [d for d in cycles_prefs.devices if d.type == backend]
            if not gpu_devices:
                continue

            # Enable all visible GPUs (CUDA_VISIBLE_DEVICES isolates per-GPU workers unless --pool-gpus)
            for dev in cycles_prefs.devices:
                if dev.type == backend:
                    dev.use = True
                    print(f"[BlendColab] Enabled {backend} device: {dev.name}")
                else:
                    dev.use = False

            activated_backend = backend
            break
        except Exception as exc:
            print(f"[BlendColab] Backend {backend} initialization failed: {exc}")

    if activated_backend is None:
        print("[BlendColab] WARNING: No GPU backend available; falling back to CPU rendering.")
        scene.cycles.device = "CPU"

    # Configure Denoising cleanly for both T4 and P100
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except Exception:
        pass

    # Keep scene BVH in VRAM across frames if memory allows
    scene.render.use_persistent_data = True

    if samples is not None and samples > 0:
        scene.cycles.samples = samples
        print(f"[BlendColab] Overrode Cycles samples -> {samples}")


if __name__ == "__main__":
    opts = parse_args()
    configure_cycles(samples=opts.samples, pool_gpus=opts.pool_gpus)