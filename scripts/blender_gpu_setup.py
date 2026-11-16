"""
Injected Blender startup script (`blender -b scene.blend -P blender_gpu_setup.py -- ...`).

Compatible across Blender 3.6 LTS, 4.0–4.5 LTS, and 5.0–5.3.
Auto-detects NVIDIA GPU architecture (Turing/Ampere+ vs Pascal P100), selects OPTIX
when RT cores are present with automatic fallback to CUDA, activates OpenImageDenoise
(including GPU denoising on Blender 4.1+), and applies CLI sample/multi-GPU overrides.
"""

import argparse
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
    version_str = getattr(bpy.app, "version_string", "unknown")
    print(f"[BlendColab] Configuring Cycles on Blender {version_str}")

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

    # Configure OpenImageDenoise across Blender 3.6 LTS, 4.x, and 5.x
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except Exception:
        pass

    # Enable GPU-accelerated OIDN when running on Blender 4.1+
    if hasattr(scene.cycles, "denoising_use_gpu"):
        try:
            scene.cycles.denoising_use_gpu = activated_backend is not None
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