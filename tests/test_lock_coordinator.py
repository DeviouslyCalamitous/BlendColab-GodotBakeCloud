"""
Unit and concurrency tests for DistributedLockManager.
Validates multi-worker race resolution, heartbeat liveness, and stale lock recovery.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from render_worker import DistributedLockManager, LockRecord  # noqa: E402


def test_single_worker_claim_and_complete(tmp_path: Path) -> None:
    mgr = DistributedLockManager(
        queue_dir=tmp_path,
        worker_id="colab_gpu0_aaa111",
        platform="colab",
        gpu_name="Tesla T4",
        sync_verify_delay_sec=0.05,
    )

    assert mgr.try_claim("frame_0001") is True
    assert (tmp_path / "frame_0001.lock").exists()
    assert mgr.is_done("frame_0001") is False

    mgr.mark_done("frame_0001", {"file": "frame_0001.png", "duration_sec": 12.4})
    assert not (tmp_path / "frame_0001.lock").exists()
    assert mgr.is_done("frame_0001") is True

    # Already completed tasks cannot be re-claimed
    assert mgr.try_claim("frame_0001") is False


def test_concurrent_workers_only_one_wins_lock_race(tmp_path: Path) -> None:
    """Simulates 4 Colab/Kaggle tabs racing simultaneously for the same frame."""
    num_workers = 4

    def attempt_claim(idx: int) -> bool:
        worker = DistributedLockManager(
            queue_dir=tmp_path,
            worker_id=f"worker_{idx}",
            platform="kaggle",
            gpu_name="Tesla T4",
            sync_verify_delay_sec=0.15,
        )
        return worker.try_claim("frame_0042")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        outcomes = list(pool.map(attempt_claim, range(num_workers)))

    assert sum(outcomes) == 1, f"Expected exactly 1 winner, got {outcomes}"


def test_stale_lock_recovery_after_crash(tmp_path: Path) -> None:
    """A healthy worker must reclaim a lock whose heartbeat expired."""
    dead_lock = LockRecord(
        worker_id="colab_crashed_999",
        platform="colab",
        gpu_name="Tesla P100",
        task_id="frame_0007",
        claimed_at=time.time() - 500,
        heartbeat_at=time.time() - 300,
    )
    (tmp_path / "frame_0007.lock").write_text(
        json.dumps(dead_lock.__dict__), encoding="utf-8"
    )

    rescuer = DistributedLockManager(
        queue_dir=tmp_path,
        worker_id="kaggle_rescuer_111",
        platform="kaggle",
        gpu_name="Tesla T4",
        stale_timeout_sec=180,
        sync_verify_delay_sec=0.05,
    )

    assert rescuer.try_claim("frame_0007") is True
    saved = json.loads((tmp_path / "frame_0007.lock").read_text(encoding="utf-8"))
    assert saved["worker_id"] == "kaggle_rescuer_111"


def test_active_heartbeat_prevents_premature_steal(tmp_path: Path) -> None:
    """A slow frame with an active heartbeat must not be stolen by peers."""
    owner = DistributedLockManager(
        queue_dir=tmp_path,
        worker_id="owner_worker",
        platform="colab",
        gpu_name="Tesla T4",
        stale_timeout_sec=1,
        heartbeat_interval_sec=0.2,
        sync_verify_delay_sec=0.05,
    )
    peer = DistributedLockManager(
        queue_dir=tmp_path,
        worker_id="peer_worker",
        platform="colab",
        gpu_name="Tesla T4",
        stale_timeout_sec=1,
        sync_verify_delay_sec=0.05,
    )

    assert owner.try_claim("bake_level1") is True
    stop_event, thread = owner.start_heartbeat("bake_level1")
    try:
        time.sleep(1.25)
        assert peer.try_claim("bake_level1") is False
    finally:
        stop_event.set()
        thread.join(timeout=2)


def test_release_lock_ignores_non_owner(tmp_path: Path) -> None:
    owner = DistributedLockManager(
        queue_dir=tmp_path,
        worker_id="worker_a",
        platform="colab",
        gpu_name="Tesla T4",
        sync_verify_delay_sec=0.05,
    )
    intruder = DistributedLockManager(
        queue_dir=tmp_path,
        worker_id="worker_b",
        platform="colab",
        gpu_name="Tesla T4",
        sync_verify_delay_sec=0.05,
    )

    assert owner.try_claim("frame_0010") is True
    intruder.release_lock("frame_0010")
    assert (tmp_path / "frame_0010.lock").exists()

    owner.release_lock("frame_0010")
    assert not (tmp_path / "frame_0010.lock").exists()