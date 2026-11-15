#!/usr/bin/env python3
"""
BlendColab / GodotBakeCloud — Cloud Environment Bootstrapper & Drive Binary Cache.

Responsibilities:
1. Detects Google Colab vs. Kaggle runtime environments.
2. Mounts Google Drive (native FUSE on Colab; rclone FUSE daemon via Kaggle Secrets on Kaggle).
3. Installs missing Linux system dependencies (xvfb, libvulkan1, ffmpeg, rclone).
4. Downloads and caches Blender LTS & Godot 4 binaries on Google Drive so subsequent
   cloud sessions extract directly to fast local NVMe storage in under 15 seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EnvBootstrap")

# Official LTS / Stable release defaults (override via CLI flags if needed)
DEFAULT_BLENDER_URL = (
    "https://download.blender.org/release/Blender4.2/blender-4.2.4-linux-x64.tar.xz"
)
DEFAULT_GODOT_URL = (
    "https://github.com/godotengine/godot/releases/download/4.3-stable/"
    "Godot_v4.3-stable_linux.x86_64.zip"
)


def detect_platform() -> str:
    if "COLAB_GPU" in os.environ or os.path.exists("/content"):
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def get_local_tools_dir(platform: str) -> Path:
    if platform == "colab":
        return Path("/content/cloud_farm_tools")
    if platform == "kaggle":
        return Path("/kaggle/working/cloud_farm_tools")
    return Path.home() / ".cache" / "cloud_farm_tools"


# ---------------------------------------------------------------------------
# Storage Mounting (Colab Native Drive vs. Kaggle rclone FUSE)
# ---------------------------------------------------------------------------


def mount_google_drive(platform: str, mount_point: Path) -> Path:
    """
    Mounts Google Drive and returns the root `MyDrive` path.
    - Colab: Uses `google.colab.drive.mount`
    - Kaggle: Reads `RCLONE_CONF` from Kaggle Secrets (or env) and starts `rclone mount`
    """
    if platform == "colab":
        mydrive = mount_point / "MyDrive"
        if not mydrive.exists():
            from google.colab import drive  # type: ignore

            log.info("Mounting Google Drive on Colab at %s...", mount_point)
            drive.mount(str(mount_point), force_remount=False)
        return mydrive

    if platform == "kaggle":
        mydrive = mount_point / "MyDrive"
        if mydrive.exists() and any(mydrive.iterdir()):
            log.info("Google Drive already mounted at %s", mydrive)
            return mydrive

        ensure_system_packages(["rclone", "fuse3"])
        rclone_conf_text = os.environ.get("RCLONE_CONF", "").strip()
        if not rclone_conf_text:
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore

                secrets = UserSecretsClient()
                rclone_conf_text = secrets.get_secret("RCLONE_CONF").strip()
            except Exception as exc:
                raise RuntimeError(
                    "Kaggle detected, but `RCLONE_CONF` secret was not found. "
                    "Add your rclone Google Drive config under Kaggle -> Add-ons -> Secrets."
                ) from exc

        rclone_dir = Path.home() / ".config" / "rclone"
        rclone_dir.mkdir(parents=True, exist_ok=True)
        (rclone_dir / "rclone.conf").write_text(rclone_conf_text + "\n", encoding="utf-8")

        mydrive.mkdir(parents=True, exist_ok=True)
        log.info("Starting rclone FUSE mount for Google Drive at %s...", mydrive)
        subprocess.Popen(
            [
                "rclone",
                "mount",
                "gdrive:",
                str(mydrive),
                "--vfs-cache-mode",
                "writes",
                "--dir-cache-time",
                "2s",
                "--vfs-write-back",
                "1s",
                "--allow-other",
                "--daemon",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait up to 15s for FUSE mount to become responsive
        for _ in range(30):
            if os.path.ismount(str(mydrive)):
                log.info("Kaggle Google Drive mounted successfully.")
                return mydrive
            time.sleep(0.5)

        raise RuntimeError(f"Timed out waiting for rclone mount at {mydrive}")

    # Local fallback
    mount_point.mkdir(parents=True, exist_ok=True)
    return mount_point


# ---------------------------------------------------------------------------
# System Package Installer (Fast No-Op on Warm Containers)
# ---------------------------------------------------------------------------


def ensure_system_packages(packages: list[str]) -> None:
    missing: list[str] = []
    binary_checks = {
        "xvfb": "xvfb-run",
        "ffmpeg": "ffmpeg",
        "rclone": "rclone",
        "unzip": "unzip",
    }
    for pkg in packages:
        bin_name = binary_checks.get(pkg)
        if bin_name and shutil.which(bin_name):
            continue
        if pkg == "libvulkan1" and Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1").exists():
            continue
        missing.append(pkg)

    if not missing:
        return

    log.info("Installing missing Linux packages: %s", ", ".join(missing))
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "--no-install-recommends", *missing],
        check=True,
    )


# ---------------------------------------------------------------------------
# Drive-Cached Binary Installer (Blender & Godot 4)
# ---------------------------------------------------------------------------


def _download_with_progress(url: str, dest: Path) -> None:
    log.info("Downloading %s...", url)
    tmp_download = dest.with_suffix(dest.suffix + ".part")

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        if block_num % 250 == 0 or downloaded >= total_size:
            sys.stdout.write(
                f"\r  -> {downloaded / (1024 * 1024):.1f} MB / "
                f"{total_size / (1024 * 1024):.1f} MB ({pct}%)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp_download, reporthook=_reporthook)
    sys.stdout.write("\n")
    shutil.move(str(tmp_download), str(dest))


def install_cached_binary(
    url: str,
    cache_dir: Path,
    extract_root: Path,
    tool_subdir: str,
    executable_glob: str,
) -> Path:
    """
    Fetches archive from Google Drive cache (downloading once if absent) and extracts
    it to fast local NVMe VM storage. Returns the resolved executable Path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_dir = extract_root / tool_subdir

    # Fast-path: already extracted in current VM session
    if target_dir.exists():
        existing_bins = sorted(target_dir.rglob(executable_glob))
        for candidate in existing_bins:
            if candidate.is_file() and not candidate.suffix == ".py":
                candidate.chmod(0o755)
                log.info("Using active local %s binary: %s", tool_subdir, candidate)
                return candidate

    archive_name = url.split("/")[-1]
    cached_archive = cache_dir / archive_name

    if not cached_archive.exists():
        local_tmp = Path("/tmp") / archive_name
        _download_with_progress(url, local_tmp)
        log.info("Caching archive to Google Drive: %s", cached_archive)
        shutil.copy2(local_tmp, cached_archive)
        local_tmp.unlink(missing_ok=True)
    else:
        log.info("Cache hit on Google Drive: %s", cached_archive.name)

    t0 = time.perf_counter()
    target_dir.mkdir(parents=True, exist_ok=True)

    if archive_name.endswith((".tar.xz", ".tar.gz", ".tgz")):
        subprocess.run(
            ["tar", "-xf", str(cached_archive), "-C", str(target_dir)],
            check=True,
        )
    elif archive_name.endswith(".zip"):
        ensure_system_packages(["unzip"])
        subprocess.run(
            ["unzip", "-q", "-o", str(cached_archive), "-d", str(target_dir)],
            check=True,
        )
    else:
        raise ValueError(f"Unsupported archive format: {archive_name}")

    elapsed = time.perf_counter() - t0
    matches = sorted(target_dir.rglob(executable_glob))
    binaries = [m for m in matches if m.is_file() and m.suffix not in {".py", ".txt", ".md"}]
    if not binaries:
        raise FileNotFoundError(
            f"Extracted {archive_name}, but no binary matched '{executable_glob}' in {target_dir}"
        )

    bin_path = binaries[0]
    bin_path.chmod(0o755)
    log.info("Extracted %s in %.1fs -> %s", tool_subdir, elapsed, bin_path)
    return bin_path


# ---------------------------------------------------------------------------
# Godot Addon Auto-Injector
# ---------------------------------------------------------------------------


def inject_godot_addon(repo_dir: Path, project_dir: Path) -> None:
    """
    Copies `addons/godot_bake_cloud` into the user's Godot 4 project on Drive and
    registers it in `project.godot` so automated LightmapGI baking works out-of-the-box.
    """
    src_addon = repo_dir / "godot_addon" / "addons" / "godot_bake_cloud"
    if not src_addon.exists():
        raise FileNotFoundError(f"Addon source missing at {src_addon}")

    project_godot = project_dir / "project.godot"
    if not project_godot.exists():
        raise FileNotFoundError(f"Not a valid Godot project (missing project.godot): {project_dir}")

    dest_addon = project_dir / "addons" / "godot_bake_cloud"
    dest_addon.parent.mkdir(parents=True, exist_ok=True)
    if dest_addon.exists():
        shutil.rmtree(dest_addon)
    shutil.copytree(src_addon, dest_addon)

    # Enable plugin inside project.godot
    plugin_cfg_res = "res://addons/godot_bake_cloud/plugin.cfg"
    text = project_godot.read_text(encoding="utf-8")

    if plugin_cfg_res not in text:
        if "[editor_plugins]" in text:
            lines = text.splitlines()
            updated_lines: list[str] = []
            in_section = False
            patched = False
            for line in lines:
                if line.strip().startswith("["):
                    if in_section and not patched:
                        updated_lines.append(f'enabled=PackedStringArray("{plugin_cfg_res}")')
                        patched = True
                    in_section = line.strip() == "[editor_plugins]"
                if in_section and line.strip().startswith("enabled=PackedStringArray("):
                    inner = line.strip()[len("enabled=PackedStringArray(") : -1].strip()
                    if inner:
                        line = f'enabled=PackedStringArray({inner}, "{plugin_cfg_res}")'
                    else:
                        line = f'enabled=PackedStringArray("{plugin_cfg_res}")'
                    patched = True
                updated_lines.append(line)
            if in_section and not patched:
                updated_lines.append(f'enabled=PackedStringArray("{plugin_cfg_res}")')
            text = "\n".join(updated_lines) + "\n"
        else:
            text = text.rstrip() + f'\n\n[editor_plugins]\n\nenabled=PackedStringArray("{plugin_cfg_res}")\n'

        project_godot.write_text(text, encoding="utf-8")
        log.info("Registered godot_bake_cloud plugin in %s", project_godot)
    else:
        log.info("godot_bake_cloud plugin already enabled in %s", project_godot)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Cloud GPU Render/Bake VM")
    parser.add_argument("--engine", choices=["blender", "godot", "both"], default="both")
    parser.add_argument("--workspace-name", default="BlendColab_Farm", help="Folder inside MyDrive")
    parser.add_argument("--repo-dir", default=".", help="Path to cloned BlendColab repository")
    parser.add_argument("--godot-project", default="", help="Optional Godot project path to inject addon")
    parser.add_argument("--blender-url", default=DEFAULT_BLENDER_URL)
    parser.add_argument("--godot-url", default=DEFAULT_GODOT_URL)
    args = parser.parse_args()

    platform = detect_platform()
    default_mount = Path("/content/drive") if platform == "colab" else Path("/kaggle/working/drive")
    mydrive = mount_google_drive(platform, default_mount)

    workspace = mydrive / args.workspace_name
    for sub in ("cache", "projects", "render_queue", "output"):
        (workspace / sub).mkdir(parents=True, exist_ok=True)

    # Install lightweight OS dependencies
    deps = ["ffmpeg"]
    if args.engine in ("godot", "both"):
        deps.extend(["xvfb", "libvulkan1"])
    ensure_system_packages(deps)

    tools_dir = get_local_tools_dir(platform)
    cache_dir = workspace / "cache"
    resolved: dict[str, str] = {
        "platform": platform,
        "workspace": str(workspace),
    }

    if args.engine in ("blender", "both"):
        blender_bin = install_cached_binary(
            url=args.blender_url,
            cache_dir=cache_dir,
            extract_root=tools_dir,
            tool_subdir="blender",
            executable_glob="blender",
        )
        resolved["blender_bin"] = str(blender_bin)

    if args.engine in ("godot", "both"):
        godot_bin = install_cached_binary(
            url=args.godot_url,
            cache_dir=cache_dir,
            extract_root=tools_dir,
            tool_subdir="godot",
            executable_glob="Godot_v4*",
        )
        resolved["godot_bin"] = str(godot_bin)

    if args.godot_project:
        inject_godot_addon(Path(args.repo_dir).resolve(), Path(args.godot_project).resolve())

    manifest_path = Path("/tmp/cloud_farm_paths.json")
    manifest_path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    log.info("Bootstrap complete. Paths saved to %s:\n%s", manifest_path, json.dumps(resolved, indent=2))


if __name__ == "__main__":
    main()