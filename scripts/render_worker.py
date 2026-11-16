#!/usr/bin/env python3
"""
BlendColab / GodotBakeCloud — Distributed Cloud GPU Render & Bake Orchestrator.

Coordinates multiple Google Colab and Kaggle GPU workers (T4, P100, T4 x2) over a
shared Google Drive directory using jitter-verified lockfiles, background heartbeats,
stale-lock recovery, automatic FFmpeg video assembly, and live Discord webhooks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

try:
    from PIL import Image
except ImportError:
    Image = None  # Falls back to ffmpeg thumbnail downscaling if Pillow is absent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("CloudFarmWorker")


# ---------------------------------------------------------------------------
# Data Models & Hardware Detection
# ---------------------------------------------------------------------------


@dataclass
class GPUDevice:
    index: int
    name: str
    vram_mb: int
    compute_cap: str
    utilization_pct: int = 0
    temp_c: int = 0

    @property
    def supports_optix(self) -> bool:
        """Turing (7.5+) and newer architectures have RT cores suited for OptiX."""
        try:
            major, minor = map(int, self.compute_cap.split(".")[:2])
            return (major, minor) >= (7, 5)
        except ValueError:
            return any(
                tag in self.name.upper() for tag in ("T4", "L4", "A100", "RTX")
            )


@dataclass
class LockRecord:
    worker_id: str
    platform: str
    gpu_name: str
    task_id: str
    claimed_at: float
    heartbeat_at: float


def detect_cloud_platform() -> str:
    if "COLAB_GPU" in os.environ or os.path.exists("/content"):
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def query_gpus() -> list[GPUDevice]:
    """Queries NVIDIA GPUs via nvidia-smi, respecting CUDA_VISIBLE_DEVICES."""
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    allowed_indices: set[int] | None = None
    if visible_env and visible_env.lower() != "all":
        allowed_indices = {
            int(idx.strip()) for idx in visible_env.split(",") if idx.strip().isdigit()
        }

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        log.warning("nvidia-smi unavailable; falling back to CPU execution mode.")
        return []

    devices: list[GPUDevice] = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        if allowed_indices is not None and idx not in allowed_indices:
            continue
        devices.append(
            GPUDevice(
                index=idx,
                name=parts[1],
                vram_mb=int(float(parts[2])),
                compute_cap=parts[3],
                utilization_pct=int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0,
                temp_c=int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 0,
            )
        )
    return devices


# ---------------------------------------------------------------------------
# Distributed Drive Lock Manager (Jitter-Verified + Heartbeat + Stale Rescue)
# ---------------------------------------------------------------------------


class DistributedLockManager:
    """
    Coordinates distributed workers across Google Drive FUSE / rclone mounts.
    Mitigates eventual-consistency race conditions via randomized backoff and
    post-write read-back ownership verification.
    """

    def __init__(
        self,
        queue_dir: Path,
        worker_id: str,
        platform: str,
        gpu_name: str,
        stale_timeout_sec: int = 240,
        heartbeat_interval_sec: int = 30,
        sync_verify_delay_sec: float = 2.5,
    ) -> None:
        self.queue_dir = queue_dir
        self.worker_id = worker_id
        self.platform = platform
        self.gpu_name = gpu_name
        self.stale_timeout_sec = stale_timeout_sec
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.sync_verify_delay_sec = sync_verify_delay_sec
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def _lock_path(self, task_id: str) -> Path:
        return self.queue_dir / f"{task_id}.lock"

    def _done_path(self, task_id: str) -> Path:
        return self.queue_dir / f"{task_id}.done"

    def _read_lock(self, lock_file: Path) -> LockRecord | None:
        try:
            data = json.loads(lock_file.read_text(encoding="utf-8"))
            return LockRecord(**data)
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def _write_lock(self, lock_file: Path, record: LockRecord) -> bool:
        try:
            tmp_file = lock_file.with_suffix(f".{self.worker_id}.tmp")
            tmp_file.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
            os.replace(tmp_file, lock_file)
            return True
        except OSError as exc:
            log.debug("Transient Drive I/O error writing lock %s: %s", lock_file.name, exc)
            return False

    def is_done(self, task_id: str) -> bool:
        return self._done_path(task_id).exists()

    def count_completed(self, task_ids: list[str]) -> int:
        return sum(1 for tid in task_ids if self.is_done(tid))

    def try_claim(self, task_id: str) -> bool:
        """
        Attempts to claim `task_id`. Returns True only if the task is not done,
        not actively locked by a healthy peer, and survives read-back verification.
        """
        if self.is_done(task_id):
            return False

        lock_file = self._lock_path(task_id)
        now = time.time()

        if lock_file.exists():
            existing = self._read_lock(lock_file)
            if existing is not None:
                age = now - existing.heartbeat_at
                if age < self.stale_timeout_sec:
                    return False
                log.warning(
                    "Reclaiming stale lock '%s' abandoned by %s (silent for %.0fs).",
                    task_id,
                    existing.worker_id,
                    age,
                )

        # Jitter reduces simultaneous write collisions across multi-tab workers
        time.sleep(random.uniform(0.2, 2.2))

        # Re-check after jitter sleep
        if self.is_done(task_id):
            return False
        if lock_file.exists():
            existing = self._read_lock(lock_file)
            if existing and (time.time() - existing.heartbeat_at) < self.stale_timeout_sec:
                return False

        now = time.time()
        record = LockRecord(
            worker_id=self.worker_id,
            platform=self.platform,
            gpu_name=self.gpu_name,
            task_id=task_id,
            claimed_at=now,
            heartbeat_at=now,
        )
        if not self._write_lock(lock_file, record):
            return False

        # Wait for Drive FUSE / rclone metadata sync and verify ownership
        time.sleep(self.sync_verify_delay_sec)
        verified = self._read_lock(lock_file)
        if verified is None or verified.worker_id != self.worker_id:
            log.info("Lost lock race on '%s' to peer worker; yielding.", task_id)
            return False

        return True

    def start_heartbeat(self, task_id: str) -> tuple[threading.Event, threading.Thread]:
        stop_event = threading.Event()
        lock_file = self._lock_path(task_id)

        def _beat() -> None:
            while not stop_event.wait(self.heartbeat_interval_sec):
                current = self._read_lock(lock_file)
                if current and current.worker_id == self.worker_id:
                    current.heartbeat_at = time.time()
                    self._write_lock(lock_file, current)

        thread = threading.Thread(target=_beat, name=f"hb-{task_id}", daemon=True)
        thread.start()
        return stop_event, thread

    def mark_done(self, task_id: str, metadata: dict[str, Any]) -> None:
        done_file = self._done_path(task_id)
        lock_file = self._lock_path(task_id)
        payload = {
            "task_id": task_id,
            "worker_id": self.worker_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        done_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass

    def release_lock(self, task_id: str) -> None:
        lock_file = self._lock_path(task_id)
        current = self._read_lock(lock_file)
        if current and current.worker_id == self.worker_id:
            try:
                lock_file.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Discord Webhook Telemetry & Thumbnail Preview Generator
# ---------------------------------------------------------------------------


class DiscordTelemetry:
    def __init__(self, webhook_url: str | None) -> None:
        self.webhook_url = (webhook_url or "").strip()

    @property
    def enabled(self) -> bool:
        return self.webhook_url.startswith("https://")

    @staticmethod
    def _create_thumbnail(image_path: Path, max_width: int = 640) -> Path | None:
        if not image_path or not image_path.exists():
            return None
        thumb_path = Path("/tmp") / f"thumb_{uuid.uuid4().hex[:8]}.jpg"

        if Image is not None:
            try:
                with Image.open(image_path) as img:
                    rgb = img.convert("RGB")
                    if rgb.width > max_width:
                        ratio = max_width / float(rgb.width)
                        rgb = rgb.resize((max_width, int(rgb.height * ratio)))
                    rgb.save(thumb_path, format="JPEG", quality=82)
                return thumb_path
            except Exception as exc:
                log.debug("Pillow thumbnail failed (%s), trying ffmpeg.", exc)

        # Fallback to ffmpeg if Pillow is unavailable or image is EXR/HDR
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(image_path),
                    "-vf",
                    f"scale='min({max_width},iw)':-2",
                    "-q:v",
                    "4",
                    str(thumb_path),
                ],
                check=True,
                timeout=15,
            )
            return thumb_path if thumb_path.exists() else None
        except Exception:
            return None

    def send_update(
        self,
        title: str,
        description: str,
        color: int = 0x5865F2,
        fields: list[dict[str, Any]] | None = None,
        preview_image: Path | None = None,
    ) -> None:
        if not self.enabled:
            return

        embed: dict[str, Any] = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "BlendColab / GodotBakeCloud"},
        }

        thumb = self._create_thumbnail(preview_image) if preview_image else None
        try:
            if thumb and thumb.exists():
                embed["image"] = {"url": f"attachment://{thumb.name}"}
                self._post_multipart({"embeds": [embed]}, thumb)
            else:
                self._post_json({"embeds": [embed]})
        except Exception as exc:
            log.warning("Discord webhook notification failed: %s", exc)
        finally:
            if thumb and thumb.exists():
                thumb.unlink(missing_ok=True)

    def _post_json(self, payload: dict[str, Any]) -> None:
        req = urllib.request.Request(
            self.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "CloudBakeWorker/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15):
            pass

    def _post_multipart(self, payload: dict[str, Any], file_path: Path) -> None:
        boundary = f"----CloudFarmBoundary{uuid.uuid4().hex}"
        body = bytearray()

        body.extend(f"--{boundary}\r\n".encode())
        body.extend(b'Content-Disposition: form-data; name="payload_json"\r\n')
        body.extend(b"Content-Type: application/json\r\n\r\n")
        body.extend(json.dumps(payload).encode("utf-8"))
        body.extend(b"\r\n")

        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="files[0]"; filename="{file_path.name}"\r\n'.encode()
        )
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())

        req = urllib.request.Request(
            self.webhook_url,
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "CloudBakeWorker/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20):
            pass


# ---------------------------------------------------------------------------
# Engine Executors (Blender Cycles & Godot 4 LightmapGI)
# ---------------------------------------------------------------------------


def run_blender_frame(
    blender_bin: Path,
    blend_file: Path,
    setup_script: Path,
    output_dir: Path,
    frame: int,
    samples: int | None,
    pool_gpus: bool,
) -> Path:
    output_pattern = output_dir / "frame_####"
    cmd = [
        str(blender_bin),
        "-b",
        str(blend_file),
        "-P",
        str(setup_script),
        "-o",
        str(output_pattern),
        "-f",
        str(frame),
        "--",
    ]
    if samples is not None:
        cmd.extend(["--samples", str(samples)])
    if pool_gpus:
        cmd.append("--pool-gpus")

    log.info("Executing Blender frame %04d...", frame)
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Blender exited with non-zero status {proc.returncode} on frame {frame}")

    # Locate rendered output file (supports PNG, EXR, JPEG depending on .blend settings)
    matches = sorted(output_dir.glob(f"frame_{frame:04d}.*"))
    if not matches:
        raise FileNotFoundError(f"Blender reported success, but no output file found for frame {frame:04d}")
    return matches[0]


def assemble_video_ffmpeg(output_dir: Path, start_frame: int, fps: int = 24) -> Path | None:
    """Assembles sequential PNG/JPG frames into an H.264 MP4 once all frames complete."""
    mp4_path = output_dir / "final_render.mp4"
    lock_assembly = output_dir / ".ffmpeg_assembly.lock"
    if mp4_path.exists() or lock_assembly.exists():
        return mp4_path if mp4_path.exists() else None

    try:
        lock_assembly.write_text(str(os.getpid()), encoding="utf-8")
        # Detect extension from start_frame
        sample_files = list(output_dir.glob(f"frame_{start_frame:04d}.*"))
        if not sample_files:
            return None
        ext = sample_files[0].suffix
        if ext.lower() == ".exr":
            log.info("Output format is EXR; skipping automatic MP4 assembly.")
            return None

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-framerate",
            str(fps),
            "-start_number",
            str(start_frame),
            "-i",
            str(output_dir / f"frame_%04d{ext}"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(mp4_path),
        ]
        subprocess.run(cmd, check=True)
        log.info("Assembled final video: %s", mp4_path)
        return mp4_path
    except Exception as exc:
        log.warning("FFmpeg assembly failed: %s", exc)
        return None
    finally:
        lock_assembly.unlink(missing_ok=True)


def run_godot_bake(
    godot_bin: Path,
    project_dir: Path,
    scene_res_path: str,
    report_path: Path,
) -> dict[str, Any]:
    """
    Launches Godot 4 Editor under Xvfb + Vulkan ICD to trigger LightmapGI baking
    via the bundled `godot_bake_cloud` EditorPlugin.
    """
    if report_path.exists():
        report_path.unlink()

    env = os.environ.copy()
    env.setdefault("DISPLAY", ":99")

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
    log.info("Baking Godot scene '%s' under Xvfb + Vulkan...", scene_res_path)
    proc = subprocess.run(cmd, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Godot exited with code {proc.returncode} while baking {scene_res_path}")

    if not report_path.exists():
        raise RuntimeError(f"Godot finished without writing bake report at {report_path}")

    return json.loads(report_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main Orchestration Loop & Multi-GPU Supervisor
# ---------------------------------------------------------------------------


def disconnect_runtime(platform: str) -> None:
    log.info("Auto-disconnect requested to preserve free GPU quota...")
    time.sleep(3)
    if platform == "colab":
        try:
            from google.colab import runtime  # type: ignore

            runtime.unassign()
        except Exception as exc:
            log.warning("Colab runtime.unassign() unavailable: %s", exc)
    os._exit(0)


def worker_loop(args: argparse.Namespace) -> None:
    platform = detect_cloud_platform()
    gpus = query_gpus()
    primary_gpu = gpus[0] if gpus else GPUDevice(0, "CPU-Fallback", 0, "0.0")
    worker_id = f"{platform}_gpu{primary_gpu.index}_{uuid.uuid4().hex[:6]}"

    workspace = Path(args.workspace).resolve()
    queue_dir = workspace / "render_queue"
    output_dir = workspace / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    stale_timeout = 180 if args.engine == "blender" else 600
    locks = DistributedLockManager(
        queue_dir=queue_dir,
        worker_id=worker_id,
        platform=platform,
        gpu_name=primary_gpu.name,
        stale_timeout_sec=stale_timeout,
    )
    discord = DiscordTelemetry(args.discord_webhook)

    # Build task list
    if args.engine == "blender":
        tasks = [f"frame_{f:04d}" for f in range(args.start_frame, args.end_frame + 1)]
    else:
        scenes = [s.strip() for s in args.godot_scenes.split(",") if s.strip()]
        tasks = [
            "bake_" + s.replace("res://", "").replace("/", "_").replace(".tscn", "")
            for s in scenes
        ]
        scene_lookup = dict(zip(tasks, scenes))

    log.info(
        "Worker %s online | Platform: %s | GPU: %s (%d MB, CC %s) | Tasks: %d",
        worker_id,
        platform.upper(),
        primary_gpu.name,
        primary_gpu.vram_mb,
        primary_gpu.compute_cap,
        len(tasks),
    )

    discord.send_update(
        title=f"Worker Joined: `{worker_id}`",
        description=f"Connected on **{platform.upper()}** and scanning `{args.engine}` queue.",
        color=0x2ECC71,
        fields=[
            {"name": "GPU", "value": f"`{primary_gpu.name}` ({primary_gpu.vram_mb} MB)", "inline": True},
            {"name": "Backend", "value": "OptiX" if primary_gpu.supports_optix else "CUDA", "inline": True},
            {"name": "Queue Size", "value": str(len(tasks)), "inline": True},
        ],
    )

    durations: list[float] = []
    scripts_dir = Path(__file__).resolve().parent
    blender_setup = scripts_dir / "blender_gpu_setup.py"

    # Graceful SIGINT/SIGTERM handling to release active lock immediately
    active_task: str | None = None

    def _handle_sigterm(signum: int, _frame: Any) -> None:
        if active_task:
            log.warning("Signal %d received; releasing lock on %s", signum, active_task)
            locks.release_lock(active_task)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    while True:
        claimed_task: str | None = None
        for task_id in tasks:
            if locks.try_claim(task_id):
                claimed_task = task_id
                break

        if claimed_task is None:
            completed = locks.count_completed(tasks)
            if completed >= len(tasks):
                log.info("All %d tasks completed!", len(tasks))
                if args.engine == "blender" and args.assemble_video:
                    mp4 = assemble_video_ffmpeg(output_dir, args.start_frame, args.fps)
                    if mp4:
                        discord.send_update(
                            title="Render Queue Complete + MP4 Assembled",
                            description=f"All **{len(tasks)}** frames finished and encoded to `{mp4.name}`.",
                            color=0xF1C40F,
                        )
                else:
                    discord.send_update(
                        title="Cloud Farm Queue Complete",
                        description=f"All **{len(tasks)}** `{args.engine}` tasks have finished.",
                        color=0xF1C40F,
                    )
                break

            # Remaining tasks are currently locked by active peer workers; poll gently
            log.info(
                "Progress: %d/%d done. Remaining tasks locked by peers; waiting 15s...",
                completed,
                len(tasks),
            )
            time.sleep(15)
            continue

        active_task = claimed_task
        hb_stop, hb_thread = locks.start_heartbeat(claimed_task)
        t0 = time.perf_counter()

        try:
            if args.engine == "blender":
                frame_num = int(claimed_task.split("_")[1])
                rendered_file = run_blender_frame(
                    blender_bin=Path(args.blender_bin),
                    blend_file=Path(args.blend_file),
                    setup_script=blender_setup,
                    output_dir=output_dir,
                    frame=frame_num,
                    samples=args.samples,
                    pool_gpus=args.pool_gpus,
                )
                elapsed = time.perf_counter() - t0
                durations.append(elapsed)
                locks.mark_done(
                    claimed_task,
                    {"file": rendered_file.name, "duration_sec": round(elapsed, 2)},
                )
                preview_img: Path | None = rendered_file
                extra_info = f"Saved `{rendered_file.name}`"
            else:
                scene_res = scene_lookup[claimed_task]
                report_file = Path("/tmp") / f"godot_report_{uuid.uuid4().hex[:6]}.json"
                report = run_godot_bake(
                    godot_bin=Path(args.godot_bin),
                    project_dir=Path(args.godot_project),
                    scene_res_path=scene_res,
                    report_path=report_file,
                )
                elapsed = time.perf_counter() - t0
                durations.append(elapsed)
                locks.mark_done(
                    claimed_task,
                    {"scene": scene_res, "duration_sec": round(elapsed, 2), **report},
                )
                preview_img = None
                extra_info = f"Baked `{scene_res}` -> `{report.get('lightmap_file', 'saved')}`"

            # Telemetry calculation
            completed = locks.count_completed(tasks)
            remaining = max(0, len(tasks) - completed)
            avg_sec = sum(durations) / len(durations)
            eta_min = (remaining * avg_sec) / 60.0
            live_gpu = query_gpus()
            gpu_stat = (
                f"{live_gpu[0].utilization_pct}% | {live_gpu[0].temp_c}°C"
                if live_gpu
                else "N/A"
            )

            log.info(
                "Completed %s in %.1fs (%d/%d | ETA ~%.1f min)",
                claimed_task,
                elapsed,
                completed,
                len(tasks),
                eta_min,
            )

            if completed % args.notify_every == 0 or remaining == 0:
                discord.send_update(
                    title=f"Completed `{claimed_task}` ({completed}/{len(tasks)})",
                    description=f"{extra_info} on `{worker_id}`.",
                    color=0x3498DB,
                    fields=[
                        {"name": "Task Time", "value": f"`{elapsed:.1f}s`", "inline": True},
                        {"name": "GPU Load", "value": f"`{gpu_stat}`", "inline": True},
                        {"name": "Est. Remaining", "value": f"`~{eta_min:.1f} min`", "inline": True},
                    ],
                    preview_image=preview_img,
                )

        except Exception as exc:
            log.error("Task '%s' failed on %s: %s", claimed_task, worker_id, exc)
            locks.release_lock(claimed_task)
            discord.send_update(
                title=f"Task Failed: `{claimed_task}`",
                description=f"Worker `{worker_id}` encountered an error and released the lock:\n```{exc}```",
                color=0xE74C3C,
            )
            time.sleep(5)
        finally:
            hb_stop.set()
            hb_thread.join(timeout=2)
            active_task = None

    if args.auto_disconnect:
        disconnect_runtime(platform)


def main() -> None:
    parser = argparse.ArgumentParser(description="BlendColab / GodotBakeCloud Worker")
    parser.add_argument("--engine", choices=["blender", "godot"], required=True)
    parser.add_argument("--workspace", required=True, help="Shared Drive folder path")
    parser.add_argument("--discord-webhook", default="", help="Discord webhook URL")
    parser.add_argument("--notify-every", type=int, default=1, help="Send Discord update every N tasks")
    parser.add_argument("--auto-disconnect", action="store_true", help="Disconnect runtime when queue finishes")
    parser.add_argument(
        "--pool-gpus",
        action="store_true",
        help="Pool all GPUs into one process instead of 1 worker per GPU",
    )

    # Blender options
    parser.add_argument("--blender-bin", default="/content/blender/blender")
    parser.add_argument("--blend-file", default="")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=1)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--assemble-video", action="store_true")

    # Godot options
    parser.add_argument("--godot-bin", default="/content/godot/godot")
    parser.add_argument("--godot-project", default="")
    parser.add_argument("--godot-scenes", default="res://main.tscn", help="Comma-separated res:// scene paths")

    args = parser.parse_args()

    # On Kaggle T4 x2 (or multi-GPU rigs), spawn 1 worker per GPU unless --pool-gpus is set
    gpus = query_gpus()
    if len(gpus) > 1 and not args.pool_gpus and "CUDA_VISIBLE_DEVICES" not in os.environ:
        log.info("Detected %d GPUs; launching parallel per-GPU workers.", len(gpus))
        procs: list[subprocess.Popen[str]] = []
        for gpu in gpus:
            sub_env = os.environ.copy()
            sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
            # Only the first sub-worker triggers auto-disconnect after all complete
            child_argv = [sys.executable, *sys.argv]
            if gpu.index != gpus[0].index and "--auto-disconnect" in child_argv:
                child_argv.remove("--auto-disconnect")
            procs.append(subprocess.Popen(child_argv, env=sub_env))

        for p in procs:
            p.wait()
        if args.auto_disconnect:
            disconnect_runtime(detect_cloud_platform())
        return

    worker_loop(args)


if __name__ == "__main__":
    main()