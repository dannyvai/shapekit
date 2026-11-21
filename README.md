# shapekit — Metashape-style photogrammetry for videos and photos

A from-scratch photogrammetry engine that follows Agisoft Metashape's image workflow. It covers
Add Photos / Import Video → Align Photos → Optimize Cameras → Build Depth Maps → Build Point Cloud →
Build Mesh → Build Texture → Export / Generate Report. It is image- and video-only: no LiDAR, laser
scans, DEMs or orthomosaics.

It is pure Python on top of **numpy, scipy, OpenCV and scikit-image**, plus **FFmpeg** for video.
It needs no CUDA, COLMAP or compiled extensions.

## Install (Windows, uv)

1. Install FFmpeg with `winget install Gyan.FFmpeg`, then open a new terminal.
2. In this folder, run:

```bat
uv sync
uv run python -m shapekit info projects\any   :: prints e.g. "compute: CUDA: NVIDIA GeForce RTX 5070"
```

`uv sync` installs PyTorch for CUDA 12.8 (required by RTX 50-series cards), plus LightGlue, kornia, OpenCV,
SciPy and scikit-image. Without PyTorch, shapekit still runs on the CPU with SIFT.

## Feature detectors and matchers

| `--features` | Detector / descriptor | Matcher | Notes |
|---|---|---|---|
| `aliked` (default with GPU) | ALIKED-n16 | LightGlue | Best accuracy/speed balance; the basis of top Image Matching Challenge SfM pipelines |
| `superpoint` | SuperPoint | LightGlue | Classic, very robust on texture-poor and repetitive scenes |
| `disk` | DISK | LightGlue | Many dense keypoints; good for wide baselines |
| `sift` (CPU fallback) | RootSIFT | FLANN mutual ratio test | No PyTorch needed |

All of them go through MAGSAC++ geometric verification and the same SfM and bundle adjustment.

**Pair preselection / loop closure** (`--retrieval`, `--cross`):

- Each video is matched sequentially (neighbouring keyframes).
- Retrieval then adds loop-closure pairs. `--cross N` reserves N candidates per frame from *other* videos, so
  separate recordings of the same area get linked.
- `auto` = hybrid: candidates interleaved from **DINOv2 + SALAD** (learned, robust to lighting, season and large
  viewpoint change) and **VLAD** of local features (best at finding real surface overlap).
- On area_d (3 videos, candidates verified with ALIKED + LightGlue), the share of frames with a verified
  cross-video loop closure in their top 5 was: SALAD 16/40, DINOv2 17/40, VLAD 23/40, **hybrid 25/40**.

**GPU acceleration** (automatic with CUDA):

- LightGlue matching
- Depth maps
- Bundle adjustment (full GPU solver, with local/global scheduling and checkpoint/resume)
- Texture visibility (CUDA z-buffer: identical to the CPU version, 34–65× faster; texturing a 1M-face mesh
  went from ~100 s to 32 s)
With a GPU, depth maps also run on CUDA, and the presets raise depth quality one level
(balanced uses `high`, quality uses `ultra`).

## Quick start

One command processes everything:

```bat
uv run python -m shapekit process projects\area_c --video videos\area_c\20261118_161234.mp4 videos\area_c\20261118_161654.mp4 videos\area_c\20261118_161800.mp4
```

You can also double-click `process_area.bat` and drag a folder of videos onto it.

Outputs, inside the project folder:

| File | What |
|---|---|
| `report.html` | Processing report: stats, camera calibration, per-camera error, model-vs-photo renders, **interactive 3D viewer** |
| `exports/model.obj` + `.mtl` + `_texture.jpg` | Textured mesh (Blender, MeshLab, Sketchfab, CloudCompare…) |
| `exports/model.glb` | Textured mesh, single file (web, Windows 3D Viewer, Blender) |
| `exports/point_cloud.ply` | Dense coloured point cloud with normals + confidence |
| `exports/colmap/sparse/0` | Cameras + tie points in COLMAP format, for **Gaussian Splatting / NeRF** tools |
| `project.json`, `chunk_0/` | The project (like a `.psx` + `.files` folder); you can re-run any step |

Presets: `--preset fast | balanced | quality` (default `balanced`). The table below shows what each one sets.

## Desktop GUI

```bat
gui.bat                       :: or: uv run python -m shapekit gui [PROJECT]
```

The layout follows Metashape:

- **Model view (centre):** switch between Tie points, Point cloud, Mesh shaded / colours / textured with keys `1`–`5`.
  Cameras, region and markers can be shown or hidden. Rotate with left-drag, pan with middle-drag, zoom with the wheel.
  `0` resets the view; `7`, `8` and `9` give top, front and side views.
- **Selection tools:** Rect Select (`R`) and Free-form Select (`L`); Shift adds, Ctrl subtracts.
  **Delete** (`Del`) and **Crop** (`Ctrl+Shift+C`) work on tie points, dense points or mesh faces, and the texture
  stays valid. *Tools → Gradual Selection* selects tie points by reprojection error or image count. Point clouds can
  be filtered by confidence, and small floating mesh pieces can be selected.
- **Region editing:** Edit Region (`B`) gives a box you can move, scale and rotate. It bounds the point cloud and mesh.
- **Photos pane:** thumbnails with aligned/disabled status, filters (not aligned, disabled, photos containing a
  marker), enable/disable. Double-click opens a photo.
- **Photo view:** zoom and pan. Markers are drawn green (placed), yellow (guided), blue (estimated, drag to place)
  and red (outlier). Right-click to add or place a marker, or to remove a projection.
- **Reference pane:** Detect markers, Guided placement, Import CSV, Add scale bar (select two markers), and
  Update, which fits the coordinate system. Errors are shown per marker and per scale bar.
- **Tools:** Add Marker (`M`) clicks on the model; the Ruler (`U`) measures distances in metres once scaled.
- **Workflow menu:** every processing step with Metashape-like dialogs, run in the background. The log appears in
  the Console pane.

## Markers, coded targets and scale

```bat
uv run python -m shapekit detect-markers projects\area_c                 :: circular coded targets (12/14/16/20-bit)
uv run python -m shapekit scalebar projects\area_c "target 34" "target 40" 2.135   :: known distance in metres
uv run python -m shapekit reference projects\area_c gcps.csv             :: label,x,y,z[,accuracy] for 3+ markers
uv run python -m shapekit export projects\area_c --model site.glb --markers markers.csv
```

- **Coded targets:** the detector fits the centre dot, checks the white gap, finds the code ring's radius
  automatically and reads the 12 sectors. Codes are rotation-invariant (minimum cyclic shift), and an ID is kept only
  if it triangulates consistently across at least 2 photos.
  *Numbering:* `target N` is the position of the code in the list generated by the published rule (expired patent
  DE19733466A1: even parity, at least one opposite pair of black sectors). That gives 147 codes for 12-bit. This may
  differ from the number printed on Metashape's sheets; rename markers in the Reference pane if needed.
  The raw 12-bit code is kept in the `code` column.
- **Guided placement:** after triangulation, each target is projected into every photo and snapped to the centre dot.
  This adds observations where the code was too small to read.
- **Scale bars:** one or more known distances between markers set the scale. The model keeps its orientation
  (Z up), in metres.
- **Reference coordinates** (total station / GNSS) on 3 or more markers fit a similarity transform
  (georeferencing or a site grid). Scale bars refine the scale.
- Exports (OBJ/GLB/PLY/COLMAP/markers CSV), the report and the GUI ruler all use the resulting coordinate system.
- **Markers in the bundle adjustment.** Coded targets are detected *before* alignment and enter it as tracks
  weighted 4× a tie point (0.5 px vs 1 px). A target seen in two videos ties them together even when their
  features never matched. If a video still forms a separate component, it is aligned on its own and **merged
  through ≥3 shared markers** (RANSAC similarity), then everything is jointly optimised. (Test: a clip split into
  two halves with all cross-half matches removed aligns 26/52 cameras without markers and 52/52 with them; the
  merged geometry is within 0.09% of the fully-matched reconstruction.) Disable with `align --no-markers`.
- **GCP-constrained optimisation:** `optimize` (or *Reference → Optimize with GCPs*) with ≥3 reference markers
  fits the transform, adds the surveyed positions as priors (weighted by their accuracy) in the bundle
  adjustment, re-fits and repeats. This corrects bending and doming of long sequences, not just their placement.
  Disable with `optimize --no-gcp`.

## Step by step (like the Workflow menu)

```bat
python -m shapekit add-video  P videos\a.mp4 videos\b.mp4   :: Import Video (smart keyframes)
python -m shapekit add-photos P D:\photos\site               :: Add Photos
python -m shapekit quality    P --disable-below 0.5          :: Estimate Image Quality
python -m shapekit align      P --accuracy high --keypoints 10000
python -m shapekit optimize   P --max-error 1.0              :: gradual selection + Optimize Cameras
python -m shapekit depth      P --quality medium --filter moderate
python -m shapekit cloud      P
python -m shapekit mesh       P --face-count medium
python -m shapekit texture    P --size 8192
python -m shapekit export     P --model out.glb --points out.ply --colmap out_colmap
python -m shapekit report     P
python -m shapekit info       P
python -m shapekit sensor     P --model fisheye_equidistant   :: Camera Calibration -> camera type
python -m shapekit masks      P ai --groups people vehicles     :: Generate Masks
python -m shapekit depth      P --quality high --method patchmatch
python -m shapekit close-holes P --max-edges 200
python -m shapekit refine-mesh P --iterations 3
python -m shapekit texture    P --blending multiband
python -m shapekit chunks     P align 0 1 --by markers      :: then: chunks P merge 0 1
python -m shapekit doctor                                    :: GPU / models / FFmpeg / GUI check
```

## Metashape feature map (image/video scope)

| Metashape | shapekit | Notes |
|---|---|---|
| Add Photos / Add Folder | `add-photos` | Images with the same resolution share one calibrated sensor |
| Import Video (frame step) | `add-video` | **Motion-adaptive** keyframes: optical-flow tracking takes a new frame every `--shift` of image motion and keeps the sharpest frame in each window. Handles 120 fps/VFR phone video |
| Estimate Image Quality | `quality` | Laplacian sharpness, normalised per sensor; disable below threshold |
| Align Photos: accuracy, key point limit | `align --accuracy --keypoints` | RootSIFT features at 2× / 1× / ½ / ¼ / ⅛ scale |
| Generic / Sequential preselection | automatic | VLAD global descriptors for loop closure plus capture-order neighbours for video; exhaustive matching for small sets |
| Incremental SfM + bundle adjustment | built in | Essential-matrix initial pair, PnP-RANSAC registration, multi-view DLT triangulation, Schur-complement LM bundle adjustment with Huber loss |
| Camera calibration (f, cx, cy, k1–k3, p1, p2) | self-calibrated | Brown-Conrady model, shared per sensor, refined in BA |
| Gradual Selection (reprojection error) + Optimize Cameras | `optimize --max-error` | |
| Region (bounding box) | auto | Robust PCA box around tie points, up axis towards the cameras |
| Build Depth Maps: quality, filtering | `depth --quality --filter` | Multi-view plane sweep with windowed NCC and occlusion-robust best-k aggregation; mild / moderate / aggressive geometric-consistency filtering |
| Build Point Cloud | `cloud` | Fused points with colour, normals and confidence (number of consistent views), voxel-merged |
| Build Mesh (depth maps source), face count | `mesh --face-count` | Narrow-band TSDF fusion + marching cubes, small-component filter, Taubin smoothing |
| Decimate Mesh | `export --max-faces` | Vertex-clustering decimation |
| Build Texture (generic mapping, mosaic) | `texture --size` | Best view per face (resolution × angle × sharpness), label smoothing, chart atlas, per-camera colour gain correction |
| Export Model (OBJ, PLY, GLB) / Export Points (PLY) | `export` | |
| Export Cameras | `export --colmap` | COLMAP text format (OPENCV model) |
| Generate Report | `report` | Self-contained HTML with a 3D viewer (needs internet once to load three.js) |
| Batch Process / headless / Python API | `process`, `import shapekit` | Each step is a plain Python function |

## Roadmap features (v0.6)

All of them are opt-in: with the default settings results are exactly as before.

| Metashape | shapekit | How |
|---|---|---|
| Build Depth Maps with surface normals | **PatchMatch** (`depth --method patchmatch`, GUI: Method) | Per-pixel slanted planes, bilateral-weighted NCC over 25 window samples, red-black propagation + random refinement, coarse-to-fine, CUDA. On a synthetic site with ground truth (ground seen at grazing angles + boxes) the median depth error drops from 0.80% (plane sweep) to 0.46%; 80% vs 59% of pixels within 1% |
| Refine Mesh | `refine-mesh`, Tools → Mesh → Refine Mesh | Each vertex searches along its normal for the best photo-consistency (5×5 surface patch, NCC over its 6 best photos), offsets smoothed over the mesh, 3 shrinking iterations; original kept as `mesh_unrefined.ply` |
| Close Holes | `close-holes --max-edges N`, Tools → Mesh → Close Holes | Holes up to N border edges are filled with a smooth membrane (concentric rings + harmonic relaxation); the outer border is never closed |
| Texture blending: mosaic / multiband (seam levelling) | `texture --blending multiband` | Detail from each face's best photo, low frequencies blended across all photos that see the surface. Colour steps across texture seams: 21.8 → 16.6 levels on the area_c test clip |
| Masks: import, manual, AI | `masks ai / import / clear`, Tools → Masks, photo view → *Edit mask* | Stored per photo (black = ignored). Applied to tie points, depth maps and texture. AI masks: DeepLabV3 (people, vehicles, animals). Manual: brush, free-form outline, erase, invert |
| Camera types: frame, fisheye | `sensor --model fisheye_equidistant`, Tools → Camera Calibration | Equidistant / equisolid fisheye + Brown distortion, in alignment, both bundle adjusters, undistortion for depth/mesh/texture, COLMAP export (OPENCV_FISHEYE). Test: area_c re-rendered through a fisheye lens aligns 52/52, f recovered to 0.02% |
| Chunks: add, duplicate, align, merge | `chunks add/duplicate/align/merge`, Workspace pane context menu, Workflow menu | Align by shared markers (coded targets) or shared photos; merge maps cameras, tie points, markers, depth maps (rescaled), point clouds and meshes into one chunk |
| (speed) | `align --ba-precision mixed` | GPU bundle adjustment with float32 Jacobians / Schur blocks and float64 accumulation + LM acceptance test |

## Tests and benchmarks

Run from the repo root; every `.bat` writes its log next to itself. All tests work on copies of `projects\gpu_test`.

| Folder | Script | What it checks |
|---|---|---|
| `tests\` | `roadmap_selftest.bat` | PatchMatch vs ground truth, fisheye, masks, mesh tools, multiband texture, chunks, mixed-precision BA |
| | `gui_selftest.bat` | Drives the real GUI with mouse events (selection, region, markers, masks, chunks), screenshots |
| | `feature_test.bat` | GPU rasteriser / texture visibility, SALAD retrieval |
| | `mixed_depth_test.bat` | Depth maps with portrait + landscape neighbours |
| | `gpu_test.bat` | One-time setup + end-to-end GPU run on one clip |
| | `test_and_build.bat` | Roadmap + GUI self-tests, then the portable build |
| `benchmarks\` | `speed_test.bat` | Optimised vs original code paths: timing + identical outputs |
| | `perf_test.bat` | GPU bundle adjustment + re-alignment timing |
| | `depth_bench.bat` | GPU plane-sweep kernel |
| | `retrieval_test.bat` | SALAD / DINOv2 / VLAD / hybrid loop-closure retrieval on area_d |

Python modules: `uv run python -m tests.roadmap_selftest projects\gpu_test`, `python -m benchmarks.speed_selftest ...`.
`shapekit doctor` checks an installation (GPU, model weights, FFmpeg, GUI).

## Portable build (Windows)

Double-click **`build_portable.bat`** (10–25 min, log in `build_portable.log`). It downloads every network weight
into `models\`, builds with PyInstaller and produces **`dist\ShapeKit\`**:

- `ShapeKit.exe` – the desktop app (no console window). `ShapeKit.exe <project folder>` opens a project.
- `shapekit-cli.exe` – the command line (`shapekit-cli.exe process ...`, `shapekit-cli.exe doctor`).
- `models\` – ALIKED/SuperPoint/DISK, LightGlue, DINOv2 + SALAD, DeepLabV3 weights (works offline).
- `ffmpeg\` – FFmpeg for video import (copied from your PATH).

Copy the whole `ShapeKit` folder anywhere (another PC, a USB disk); nothing needs installing. It includes the CUDA
12.8 runtime: on another PC only an NVIDIA driver (570+) is needed; without an NVIDIA GPU it runs on the CPU.
It is a folder rather than a single .exe on purpose: with PyTorch + CUDA it is ~4–5 GB, and a one-file exe would
unpack all of that to a temp folder on every start.

## Tips for capturing video

- Walk slowly around the subject and keep it in frame. Sideways motion gives depth; turning on the spot does not.
- Shoot at 30 fps with a fast shutter if possible. Motion blur is the main enemy, and keyframe selection already prefers sharp frames.
- Overlapping loops at two heights (eye level, then low or high) give much more complete meshes.
- Several videos of the same area can go in one project. They are matched together automatically.

## Performance

On a 4-core CPU, a 50-frame clip takes about 1–2 minutes for features and matching, 1 minute to align, 2 minutes
for medium depth maps, and seconds to minutes for mesh and texture. Matching and depth maps use all CPU cores
(`--workers N`). High-quality depth maps cost about 6× more than medium.

The model has arbitrary scale and orientation, as in Metashape without GCPs or scale bars.
