# Architecture & Distributed Fault-Tolerance Design

## 1. Lockfile Consensus Over Eventually-Consistent Storage
Google Drive FUSE (`/content/drive`) and `rclone` VFS mounts cache directory metadata for 2–10 seconds and do not guarantee cross-VM atomic `O_CREAT | O_EXCL` locks. To prevent two GPU workers from rendering the same frame simultaneously:

1. **Random Backoff Jitter (`0.2s–2.2s`):** Desynchronizes workers scanning the queue at the same instant.
2. **Atomic Local Rename + Read-Back Verification:** Workers write `frame_XXXX.<worker_id>.tmp`, rename it to `frame_XXXX.lock`, sleep `2.5s` to let Drive's metadata layer settle, and re-read the JSON payload. If `worker_id` does not match, the worker yields immediately.
3. **Heartbeat Liveness (`30s`):** A daemon thread updates `heartbeat_at` inside the `.lock` file while Blender or Godot is running.
4. **Stale Lock Rescue (`180s` Blender / `600s` Godot):** When a free Colab or Kaggle runtime disconnects abruptly, `SIGTERM` handlers may never execute. Healthy workers inspect existing `.lock` timestamps and reclaim expired tasks automatically.

## 2. Headless Vulkan Lightmap Baking in Godot 4
Godot 4's `LightmapGI` baker relies on Vulkan compute shaders (`RenderingDevice`) and editor UI hooks (`LightmapGIEditorPlugin`), both of which are disabled by `--headless`. `godot_bake_runner.py` launches the standard Godot 4 Linux binary inside `xvfb-run` with `--rendering-driver vulkan` and passes `--bake-scene=res://...` to our dormant-by-default `EditorPlugin`, which opens the scene, triggers the bake pipeline, awaits `.lmbake` disk flush, saves the scene, and exits cleanly.