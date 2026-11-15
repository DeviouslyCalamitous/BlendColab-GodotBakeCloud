# BlendColab / GodotBakeCloud

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeviouslyCalamitous/BlendColab-GodotBakeCloud/blob/main/notebooks/BlendColab_GodotBake.ipynb)
[![Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Blender 4.2 LTS](https://img.shields.io/badge/Blender-4.2%20LTS-F5792A?logo=blender&logoColor=white)](https://www.blender.org/)
[![Godot 4.3+](https://img.shields.io/badge/Godot-4.3%2B-478CBF?logo=godotengine&logoColor=white)](https://godotengine.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**BlendColab / GodotBakeCloud** turns free cloud GPU instances (**Google Colab** & **Kaggle** `T4`, `P100`, and `T4 x2` dual-GPU runtimes) into a fault-tolerant, distributed render and lightmap baking farm backed by Google Drive.

Open 2 to 4 browser tabs across Colab and Kaggle simultaneously—workers coordinate over a shared Google Drive lock queue, recover dropped frames automatically if a free VM disconnects, and stream rendered preview thumbnails to Discord.

---

## System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                        SHARED GOOGLE DRIVE BACKBONE                          │
│  /MyDrive/BlendColab_Farm/                                                   │
│   ├── cache/          (Cached Blender LTS & Godot 4 archives -> <15s boot)   │
│   ├── projects/       (scene.blend OR extracted Godot 4 project folder)      │
│   ├── render_queue/   (frame_0001.lock, frame_0001.done, bake_level1.lock)   │
│   └── output/         (frame_0001.png ... final_render.mp4 / *.lmbake)       │
└───────▲──────────────────────────────▲──────────────────────────────▲────────┘
        │ Drive FUSE                   │ rclone FUSE                  │
┌───────┴──────────────┐       ┌───────┴──────────────┐       ┌───────┴────────┐
│  WORKER 1 (Colab)    │       │  WORKER 2 (Kaggle)   │       │  WORKER 3      │
│  NVIDIA T4 (sm_75)   │       │  NVIDIA T4 x2        │       │  NVIDIA P100   │
│  - Cycles: OPTIX     │       │  - GPU 0: OPTIX      │       │  - Cycles: CUDA│
│  - Godot: Xvfb+Vulkan│       │  - GPU 1: OPTIX      │       │  - OIDN Denoise│
└───────┬──────────────┘       └───────┬──────────────┘       └───────┬────────┘
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │   DISCORD WEBHOOK TELEMETRY   │
                        │  Embeds + Thumbnails + ETA   │
                        └──────────────────────────────┘
```

---

## Key Engineering Highlights

| Challenge | Solution |
| :--- | :--- |
| **Drive API Eventual Consistency** | Google Drive FUSE does not guarantee distributed `O_CREAT \| O_EXCL` atomicity. Workers apply randomized backoff (`0.2s–2.2s`), write JSON lock records, wait `2.5s` for metadata propagation, and re-read the `.lock` file to verify ownership before rendering. |
| **Free VM Disconnect Recovery** | A background daemon thread updates `heartbeat_at` every `30s`. If a free Colab/Kaggle tab times out mid-frame, peer workers detect the expired lock (`>180s` Blender / `>600s` Godot) and rescue the orphaned task. |
| **T4 vs. P100 Architecture Routing** | `blender_gpu_setup.py` inspects GPU compute capability: Turing/Ampere+ GPUs (`T4`, `L4`, `A100`) activate hardware RT-core **`OPTIX`**, while Pascal GPUs (`P100`) select **`CUDA` + `OpenImageDenoise`** to avoid OptiX kernel stalls. |
| **Kaggle `T4 x2` Parallelism** | Automatically spawns two isolated worker processes pinned via `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1` to render two frames in parallel (or pools both GPUs via `--pool-gpus` for heavy single scenes). |
| **Headless Godot 4 `LightmapGI`** | Godot's `--headless` flag disables Vulkan `RenderingDevice`, and `LightmapGI.bake()` is editor-only. `godot_bake_cloud` launches Godot 4 Editor inside `xvfb-run` with NVIDIA Vulkan ICD active, locates the `LightmapGI` node, triggers the bake pipeline, and saves `.lmbake` files directly to Drive. |

---

## Quickstart

### 1. Prepare Your Project on Google Drive
1. Open [`notebooks/BlendColab_GodotBake.ipynb`](notebooks/BlendColab_GodotBake.ipynb) in **Google Colab** (or import into **Kaggle** with GPU accelerator enabled).
2. Run **Cell 1** to mount Google Drive and initialize `/MyDrive/BlendColab_Farm/`.
   - *First run:* Downloads Blender LTS & Godot 4 and caches archives in `BlendColab_Farm/cache/`.
   - *Subsequent runs:* Extracts directly from Drive cache to local NVMe in **under 15 seconds**.
3. Upload your project into `/MyDrive/BlendColab_Farm/projects/`:
   - **Blender:** Pack external textures (`File -> External Data -> Pack Resources`) and upload `scene.blend`.
   - **Godot 4:** Upload your project folder containing `project.godot` and scenes with a configured `LightmapGI` node.

### 2. Launch 1 to 4 Cloud Workers
1. Configure **Cell 2** (`ENGINE`, `PROJECT_SUBPATH`, frame range or `GODOT_SCENES`, and optional `DISCORD_WEBHOOK_URL`).
2. Run **Cell 2**. To scale horizontally, open the same notebook in additional Colab or Kaggle tabs—each worker automatically claims unclaimed tasks from `render_queue/`.
3. Enable `AUTO_DISCONNECT_WHEN_DONE = True` to automatically release your Colab runtime once the queue finishes and preserve free GPU quota.

*(For Kaggle `T4 x2` Google Drive setup, see [`docs/kaggle_rclone_setup.md`](docs/kaggle_rclone_setup.md).)*

---

## Repository Structure

```text
BlendColab-GodotBakeCloud/
├── .github/workflows/lint_and_test.yml   # Ruff + Pytest CI pipeline
├── docs/
│   ├── architecture.md                   # Distributed lock & Vulkan Xvfb deep-dive
│   └── kaggle_rclone_setup.md            # Kaggle Secrets + rclone Drive mount guide
├── godot_addon/addons/godot_bake_cloud/  # Dormant-by-default Godot 4 EditorPlugin
├── notebooks/BlendColab_GodotBake.ipynb  # 1-click Colab & Kaggle UI notebook
├── scripts/
│   ├── __init__.py
│   ├── env_bootstrap.py                  # Platform detection, Drive cache & apt setup
│   ├── render_worker.py                  # Distributed lock manager, multi-GPU & Discord telemetry
│   ├── blender_gpu_setup.py              # Cycles OptiX/CUDA + OIDN auto-configurator
│   └── godot_bake_runner.py              # Standalone Vulkan + Xvfb Godot bake CLI
└── tests/                                # Lock contention & GPU routing unit tests
```