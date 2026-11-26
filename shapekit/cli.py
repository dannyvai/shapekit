"""shapekit command line -- Metashape-style workflow.

  python -m shapekit process  PROJECT --video a.mp4 b.mp4 [--photos DIR] [--preset balanced]
  python -m shapekit new      PROJECT
  python -m shapekit add-video  PROJECT VIDEO [VIDEO ...]
  python -m shapekit add-photos PROJECT PATH  [PATH ...]
  python -m shapekit quality  PROJECT [--disable-below 0.5]
  python -m shapekit align    PROJECT [--accuracy high] [--keypoints 10000]
  python -m shapekit optimize PROJECT [--max-error 1.0]
  python -m shapekit depth    PROJECT [--quality medium] [--filter moderate]
  python -m shapekit cloud    PROJECT
  python -m shapekit mesh     PROJECT [--face-count medium]
  python -m shapekit texture  PROJECT [--size 8192]
  python -m shapekit export   PROJECT [--model out.obj|.glb|.ply] [--points out.ply] [--colmap DIR]
  python -m shapekit report   PROJECT
  python -m shapekit info     PROJECT
  python -m shapekit sensor   PROJECT --model fisheye_equidistant [--sensor 0] [--f 900] [--fixed]
  python -m shapekit masks    PROJECT ai [--groups people vehicles] | import DIR [--pattern ...] | clear
  python -m shapekit refine-mesh PROJECT [--iterations 3]
  python -m shapekit close-holes PROJECT [--max-edges 200]
  python -m shapekit flythrough PROJECT [--pattern orbit|helix|flyover|reveal|topdown|capture] [--seconds 20]
  python -m shapekit chunks   PROJECT list | add [--label L] | duplicate I | remove I
                              | align REF I [I ...] [--by markers|cameras] | merge I I [...] [--label L]
"""
import argparse
import os
import sys
import time

from .project import Project
from .util import log

PRESETS = {
    #            align acc, keypoints, depth quality, filter,     face count, texture
    "fast":     ("medium", 6000, "low", "moderate", "low", 4096),
    "balanced": ("high", 8000, "medium", "moderate", "medium", 8192),
    "quality":  ("high", 12000, "high", "mild", "high", 8192),
}
# keypoint budgets for learned features (LightGlue scales ~quadratically with keypoints)
LEARNED_KP = {"fast": 2048, "balanced": 4096, "quality": 8192}
GPU_DEPTH = {"fast": "medium", "balanced": "high", "quality": "ultra"}


def resolve_features(name):
    if name in (None, "auto"):
        from .learned import torch_available
        return "aliked" if torch_available() else "sift"
    return name


VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".webm")


def expand_videos(items):
    """Accept files, folders and wildcards (Windows shells don't expand *.mp4)."""
    import glob
    out = []
    for it in items or []:
        if os.path.isdir(it):
            out += sorted(f for f in glob.glob(os.path.join(it, "*")) if f.lower().endswith(VIDEO_EXT))
        elif any(ch in it for ch in "*?["):
            out += sorted(glob.glob(it))
        else:
            out.append(it)
    return out


def _open(a):
    p = Project.open(a.project)
    return p, p.chunk(a.chunk)


def cmd_new(a):
    Project.create(a.project); log("created %s" % a.project)


def cmd_add_video(a):
    from .video import import_video
    p = Project.open(a.project) if os.path.exists(os.path.join(a.project, Project.FILE)) else Project.create(a.project)
    ch = p.chunk(a.chunk)
    for v in expand_videos(a.videos):
        t = time.time()
        n = import_video(ch, v, overlap_shift=a.shift, max_gap_s=a.max_gap,
                             min_sharpness=getattr(a, "min_sharpness", 0.7))
        ch.add_log("Import Video", time.time() - t, video=os.path.basename(v), frames=n)
        p.save()


def cmd_add_photos(a):
    from .video import add_photos, estimate_image_quality
    p = Project.open(a.project) if os.path.exists(os.path.join(a.project, Project.FILE)) else Project.create(a.project)
    ch = p.chunk(a.chunk)
    n = add_photos(ch, a.paths)
    estimate_image_quality(ch)
    ch.add_log("Add Photos", 0, photos=n)
    p.save(); log("added %d photos" % n)


def cmd_quality(a):
    from .video import estimate_image_quality
    p, ch = _open(a)
    estimate_image_quality(ch, a.disable_below)
    off = sum(not c.enabled for c in ch.cameras)
    p.save(); log("image quality estimated; %d cameras disabled" % off)


def cmd_align(a):
    from .sfm import align_photos
    p, ch = _open(a)
    method = resolve_features(a.features)
    kp = a.keypoints or (10000 if method == "sift" else 4096)
    align_photos(ch, a.accuracy, kp, a.seq_window, a.generic, reset=a.reset, method=method,
                 reset_features=a.reset_features, retrieval=a.retrieval, cross_k=a.cross,
                 use_markers=not a.no_markers, ba_precision=a.ba_precision)
    p.save()


def cmd_optimize(a):
    from .sfm import optimize_cameras
    p, ch = _open(a)
    st = optimize_cameras(ch, max_error_px=a.max_error, gcp=not a.no_gcp, ba_precision=a.ba_precision)
    p.save(); log("optimized: RMS %.3f px" % st["rms_reprojection_px"])


def cmd_depth(a):
    from .mvs import build_depth_maps
    p, ch = _open(a)
    build_depth_maps(ch, a.quality, a.filter, force=a.reset, workers=a.workers, method=a.method)
    p.save()


def cmd_cloud(a):
    from .mvs import build_point_cloud
    p, ch = _open(a)
    build_point_cloud(ch); p.save()


def cmd_mesh(a):
    from .mesh import build_mesh
    p, ch = _open(a)
    build_mesh(ch, a.face_count, voxel_size=a.voxel); p.save()


def cmd_texture(a):
    from .texture import build_texture
    p, ch = _open(a)
    build_texture(ch, a.size, blending=a.blending); p.save()


def cmd_export(a):
    from .export import export_model, export_points, export_cameras
    p, ch = _open(a)
    if a.model:
        export_model(ch, a.model, max_faces=a.max_faces)
    if a.points:
        export_points(ch, a.points, "sparse" if a.sparse else "dense")
    if a.colmap:
        export_cameras(ch, a.colmap)
    if a.markers:
        from .export import export_markers
        export_markers(ch, a.markers)


def cmd_report(a):
    from .report import generate_report
    p, ch = _open(a)
    generate_report(ch, a.out)


def cmd_detect_markers(a):
    from .markers import detect_markers, refine_marker_projections
    p, ch = _open(a)
    if a.reset:
        ch.markers = [m for m in ch.markers if not m.get("auto")]
    detect_markers(ch, bits=a.bits, min_radius=a.min_radius, tolerance=a.tolerance)
    if not a.no_guided and ch.aligned_cameras():
        refine_marker_projections(ch)
    for m in ch.markers:
        print("  %-14s %3d projections  %s" % (m["label"], len(m["projections"]),
                                               "err %.2f px" % m["error_px"] if m.get("error_px") else "not triangulated"))
    p.save()


def cmd_scalebar(a):
    from .markers import add_scalebar, update_transform
    p, ch = _open(a)
    add_scalebar(ch, a.a, a.b, a.distance, a.accuracy)
    rep = update_transform(ch)
    for b in rep["scalebars"]:
        print("  %s - %s: %.4f (estimated %.4f, error %+.4f)" % (b["a"], b["b"], b["distance"], b["estimated"], b["error"]))
    p.save()


def cmd_reference(a):
    """CSV: label,x,y,z[,accuracy] -> marker reference coordinates, then update transform."""
    import csv
    from .markers import set_reference, update_transform, find_marker
    p, ch = _open(a)
    n = 0
    for row in csv.reader(open(a.csv)):
        if not row or row[0].startswith("#") or row[0].lower() == "label":
            continue
        if find_marker(ch, row[0]) is None:
            print("  skip %s (no such marker)" % row[0]); continue
        set_reference(ch, row[0], list(map(float, row[1:4])), float(row[4]) if len(row) > 4 and row[4] else 0.005)
        n += 1
    rep = update_transform(ch)
    for m in rep["markers"]:
        print("  %-14s error %.4f" % (m["label"], m["error_total"]))
    p.save(); log("%d reference markers" % n)


def cmd_info(a):
    from .learned import describe_device, torch_available
    print("compute: %s, learned features: %s" % (describe_device(), "available" if torch_available() else "not installed"))
    p, ch = _open(a)
    print("Project %s, chunk '%s'" % (p.dir, ch.label))
    print("  cameras: %d (%d enabled, %d aligned)" % (len(ch.cameras), sum(c.enabled for c in ch.cameras),
                                                      len(ch.aligned_cameras())))
    if len(p.chunks) > 1:
        print("  chunks: " + ", ".join("%d: %s" % (k, c.label) for k, c in enumerate(p.chunks)))
    for s in ch.sensors:
        print("  sensor %s %dx%d f=%.1f (%s%s)" % (s.label, s.width, s.height, s.intr[0], s.model, ", fixed" if s.fixed else ""))
    for k, v in ch.stats.items():
        print("  %s: %s" % (k, v))


def cmd_sensor(a):
    p, ch = _open(a)
    sens = range(len(ch.sensors)) if a.sensor is None else [a.sensor]
    for s in sens:
        sen = ch.sensors[s]
        if a.model:
            sen.set_model(a.model)
        if a.f:
            sen.intr[0] = a.f
        if a.fixed is not None:
            sen.fixed = a.fixed
        print("  sensor %d %s: %s f=%.1f%s" % (s, sen.label, sen.model, sen.intr[0], " (fixed)" if sen.fixed else ""))
    p.save()


def cmd_masks(a):
    from . import masks as MS
    p, ch = _open(a)
    if a.action == "ai":
        MS.generate_ai_masks(ch, a.groups, a.dilate)
    elif a.action == "import":
        MS.import_masks(ch, a.folder, a.pattern, invert=a.invert)
    elif a.action == "clear":
        for i in MS.masked_cameras(ch):
            MS.clear_mask(ch, i)
        log("masks cleared")
    p.save()


def cmd_refine_mesh(a):
    from .mesh import refine_mesh
    p, ch = _open(a)
    refine_mesh(ch, a.iterations); p.save()


def cmd_close_holes(a):
    from .mesh import close_holes_chunk
    p, ch = _open(a)
    close_holes_chunk(ch, a.max_edges); p.save()


def cmd_chunks(a):
    from . import chunks as CK
    p = Project.open(a.project)
    if a.action == "add":
        CK.add_chunk(p, a.label)
    elif a.action == "duplicate":
        CK.duplicate_chunk(p, a.idx[0], a.label)
    elif a.action == "remove":
        CK.remove_chunk(p, a.idx[0])
    elif a.action == "align":
        CK.align_chunks(p, a.idx[0], a.idx[1:], a.by)
    elif a.action == "merge":
        CK.merge_chunks(p, a.idx, a.label or "Merged Chunk", merge_mesh=a.mesh)
    for k, c in enumerate(p.chunks):
        print("  %d: %-20s %4d cameras (%d aligned)  %s" % (k, c.label, len(c.cameras), len(c.aligned_cameras()),
                                                           "transform set" if c.transform is not None else ""))
    p.save()


def cmd_flythrough(a):
    from .flythrough import render_flythrough
    p, ch = _open(a)
    w, h = (a.size.lower().split("x") if "x" in a.size.lower() else (None, None))
    size = (int(w), int(h)) if w else a.size.lower()
    start = a.start_angle if a.start_angle == "auto" else float(a.start_angle)
    render_flythrough(ch, a.out, a.pattern, a.seconds, a.fps, size, a.elevation, a.distance, a.display, start,
                      a.crf, arc=a.arc)
    p.save()


def cmd_process(a):
    acc, kp, dq, filt, fc, tex = PRESETS[a.preset]
    if not os.path.exists(os.path.join(a.project, Project.FILE)):
        Project.create(a.project)
    p = Project.open(a.project); ch = p.chunk(a.chunk)
    t_all = time.time()
    if a.video and not ch.cameras:
        from .video import import_video
        for v in expand_videos(a.video):
            t = time.time()
            n = import_video(ch, v, overlap_shift=a.shift, max_gap_s=a.max_gap,
                             min_sharpness=getattr(a, "min_sharpness", 0.7))
            ch.add_log("Import Video", time.time() - t, video=os.path.basename(v), frames=n)
            p.save()
    if a.photos and not any(c.source is None for c in ch.cameras):
        from .video import add_photos, estimate_image_quality
        add_photos(ch, a.photos); estimate_image_quality(ch); p.save()
    from .sfm import align_photos
    from .mvs import build_depth_maps, build_point_cloud
    from .mesh import build_mesh
    from .texture import build_texture
    from .export import export_model, export_points, export_cameras
    from .report import generate_report
    method = resolve_features(a.features)
    from .learned.mvs_gpu import gpu_available
    if method != "sift":
        kp = LEARNED_KP[a.preset]
    if gpu_available():
        dq = GPU_DEPTH[a.preset]
    seq, gen = (12, 10) if method != "sift" else (10, 12)
    steps = ["Align Photos", "Build Depth Maps"] + ([] if a.no_cloud else ["Build Point Cloud"]) + ["Build Mesh"]
    steps += (["Close Holes"] if getattr(a, "close_holes", 0) else []) + (["Refine Mesh"] if getattr(a, "refine", False) else [])
    steps += ["Build Texture", "Export + Report"] + (["Fly-through Video"] if getattr(a, "flythrough", None) else [])

    from . import projlog
    resume_from = getattr(a, "resume_from", None)     # a step name (set by --resume / the GUI's crash recovery)
    if resume_from:
        log("  resuming at step '%s' (earlier steps finished before the crash)" % resume_from)
    first = steps.index(resume_from) if resume_from in steps else 0

    def stage(name):
        """False (skip) for steps that finished before the crash being resumed."""
        i = steps.index(name)
        if i < first:
            log("========== Batch step %d/%d: %s - done before the crash, skipped ==========" % (i + 1, len(steps), name))
            return False
        log("========== Batch step %d/%d: %s ==========" % (i + 1, len(steps), name))
        projlog.journal_update(stage=name)
        return True
    if stage("Align Photos"):
        if not ch.aligned_cameras():
            align_photos(ch, acc, kp, seq, gen, method=method, use_markers=not a.no_markers); p.save()
        else:
            log("  photos already aligned (%d cameras) - skipped" % len(ch.aligned_cameras()))
    if stage("Build Depth Maps"):              # depth maps already computed (depth/*.npz) are reused
        build_depth_maps(ch, dq, filt, workers=a.workers, method=getattr(a, "depth_method", "planesweep")); p.save()
    if not a.no_cloud and stage("Build Point Cloud"):
        build_point_cloud(ch); p.save()
    if stage("Build Mesh"):
        build_mesh(ch, fc); p.save()
    if getattr(a, "close_holes", 0) and stage("Close Holes"):
        from .mesh import close_holes_chunk
        close_holes_chunk(ch, a.close_holes); p.save()
    if getattr(a, "refine", False) and stage("Refine Mesh"):
        from .mesh import refine_mesh
        refine_mesh(ch); p.save()
    if stage("Build Texture"):
        build_texture(ch, tex, blending=getattr(a, "blending", "mosaic")); p.save()
    stage("Export + Report")
    out = os.path.join(p.dir, "exports")
    log("Exporting model (OBJ + GLB), point cloud, cameras and markers to %s..." % out)
    export_model(ch, os.path.join(out, "model.obj"))
    export_model(ch, os.path.join(out, "model.glb"), texture_size=4096)
    if not a.no_cloud:
        export_points(ch, os.path.join(out, "point_cloud.ply"))
    export_cameras(ch, os.path.join(out, "colmap"), copy_images=False)
    if ch.markers:
        from .export import export_markers
        export_markers(ch, os.path.join(out, "markers.csv"))
    log("Generating the HTML report (renders, statistics, 3D viewer)...")
    generate_report(ch)
    if getattr(a, "flythrough", None):
        projlog.journal_update(stage="Fly-through Video")
        log("========== Batch step %d/%d: Fly-through Video ==========" % (len(steps), len(steps)))
        from .flythrough import run_external
        try:                                   # separate process: VTK off-screen rendering
            run_external(p.dir, a.chunk, os.path.join(out, "flythrough_%s.mp4" % a.flythrough), pattern=a.flythrough)
        except Exception as e:
            log("  fly-through video skipped: %s" % e)
    log("All done in %.1f min -> %s" % ((time.time() - t_all) / 60, out))


def cmd_config(a):
    from . import config, ffmpeg
    if a.key == "ffmpeg" and (a.value or a.clear):
        try:
            ffmpeg.set_location(None if a.clear else a.value)
        except ValueError as e:
            sys.exit("error: %s" % e)
    print("settings file:", config.path())
    print(ffmpeg.describe())


def cmd_underwater(a):
    from . import underwater as UW
    p = Project.open(a.project); ch = p.chunk(a.chunk)
    s = UW.load_settings(ch)
    if a.mode:
        s["mode"] = a.mode
    if a.patch:
        lab, *xy = a.patch
        idx = next((k for k, c in enumerate(ch.cameras) if c.label == lab or os.path.basename(c.path) == lab), None)
        if idx is None:
            sys.exit("error: no photo called %s" % lab)
        s["patch"] = dict(camera=idx, label=ch.cameras[idx].label,
                          rect=UW.photo_rect_to_undistorted(ch, idx, [float(v) for v in xy]))
    if a.no_patch:
        s["patch"] = None
    if a.true_color:
        s["ref_rgb"] = [int(v) for v in a.true_color.split(",")]
    if a.neutral:
        s["ref_rgb"] = None
    if a.strength is not None:
        s["strength"] = a.strength
    s["params"] = None
    if s["mode"] != "off":
        s["params"] = UW.estimate(ch, s)
    UW.save_settings(ch, s)
    log("underwater colour: %s -> %s (rebuild the texture to apply)" % (UW.MODE_LABELS[s["mode"]], UW.settings_path(ch)))
    if a.preview:
        import cv2
        import numpy as np
        lab, out = a.preview
        idx = next((k for k, c in enumerate(ch.cameras) if c.label == lab or os.path.basename(c.path) == lab), None)
        if idx is None:
            sys.exit("error: no photo called %s" % lab)
        bef, aft = UW.preview(ch, idx, s, scale=0.5)
        cv2.imwrite(out, np.hstack([bef, aft]))
        log("preview (before | after): %s" % out)


def main(argv=None):
    import faulthandler
    faulthandler.enable()                       # native crashes print a stack instead of closing silently
    ap = argparse.ArgumentParser(prog="shapekit", description="Photogrammetry from videos and photos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def sp(name, fn, **kw):
        s = sub.add_parser(name, **kw)
        s.add_argument("project")
        s.add_argument("--chunk", type=int, default=0)
        s.set_defaults(fn=fn)
        return s

    sp("new", cmd_new)
    s = sp("add-video", cmd_add_video); s.add_argument("videos", nargs="+")
    s.add_argument("--shift", type=float, default=0.08, help="image shift (fraction of width) between keyframes")
    s.add_argument("--max-gap", type=float, default=1.5, help="max seconds between keyframes")
    s.add_argument("--min-sharpness", type=float, default=0.7,
                   help="skip motion-blurred frames: minimum sharpness relative to nearby frames (0-1, 0 = off)")
    s = sp("add-photos", cmd_add_photos); s.add_argument("paths", nargs="+")
    s = sp("quality", cmd_quality); s.add_argument("--disable-below", type=float, default=0.5)
    s = sp("align", cmd_align)
    s.add_argument("--accuracy", default="high", choices=["highest", "high", "medium", "low", "lowest"])
    s.add_argument("--keypoints", type=int, default=None, help="default 10000 (sift) / 4096 (learned)")
    s.add_argument("--features", default="auto", choices=["auto", "sift", "superpoint", "aliked", "disk"],
                   help="auto = aliked+LightGlue if PyTorch is installed, else sift")
    s.add_argument("--seq-window", type=int, default=10)
    s.add_argument("--generic", type=int, default=10, help="retrieval pairs per photo (loop closures)")
    s.add_argument("--retrieval", default="auto", choices=["auto", "salad", "dinov2", "vlad"],
                   help="global descriptor for pair preselection (auto = hybrid SALAD + VLAD when PyTorch is installed)")
    s.add_argument("--cross", type=int, default=6, help="retrieval pairs per photo from OTHER videos")
    s.add_argument("--no-markers", action="store_true",
                   help="don't detect coded targets before alignment / use markers as tie points")
    s.add_argument("--reset", action="store_true", help="discard current alignment (keeps features/matches)")
    s.add_argument("--reset-features", action="store_true", help="also recompute features and matches")
    s.add_argument("--ba-precision", default=None, choices=["double", "mixed"],
                   help="GPU bundle adjustment arithmetic (mixed = float32 Jacobians, faster)")
    s = sp("optimize", cmd_optimize); s.add_argument("--max-error", type=float, default=None)
    s.add_argument("--no-gcp", action="store_true", help="ignore marker reference coordinates (no GCP constraints)")
    s.add_argument("--ba-precision", default=None, choices=["double", "mixed"])
    s = sp("depth", cmd_depth)
    s.add_argument("--quality", default="medium", choices=["ultra", "high", "medium", "low", "lowest"])
    s.add_argument("--filter", default="moderate", choices=["none", "mild", "moderate", "aggressive"])
    s.add_argument("--method", default="planesweep", choices=["planesweep", "patchmatch"])
    s.add_argument("--workers", type=int, default=None); s.add_argument("--reset", action="store_true")
    sp("cloud", cmd_cloud)
    s = sp("mesh", cmd_mesh); s.add_argument("--face-count", default="medium", choices=["low", "medium", "high"])
    s.add_argument("--voxel", type=float, default=None)
    s = sp("texture", cmd_texture); s.add_argument("--size", type=int, default=8192)
    s.add_argument("--blending", default="mosaic", choices=["mosaic", "multiband"])
    s = sp("sensor", cmd_sensor); s.add_argument("--sensor", type=int, default=None)
    s.add_argument("--model", choices=["frame", "fisheye_equidistant", "fisheye_equisolid"])
    s.add_argument("--f", type=float, default=None, help="initial focal length in pixels")
    s.add_argument("--fixed", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    s = sp("masks", cmd_masks); s.add_argument("action", choices=["ai", "import", "clear"])
    s.add_argument("folder", nargs="?"); s.add_argument("--pattern", default="{label}_mask.png")
    s.add_argument("--invert", action="store_true")
    s.add_argument("--groups", nargs="+", default=["people", "vehicles"], choices=["people", "vehicles", "animals"])
    s.add_argument("--dilate", type=int, default=12)
    s = sp("refine-mesh", cmd_refine_mesh); s.add_argument("--iterations", type=int, default=3)
    s = sp("close-holes", cmd_close_holes); s.add_argument("--max-edges", type=int, default=200)
    s = sp("flythrough", cmd_flythrough, help="render a drone-style fly-through video (MP4) of the model")
    s.add_argument("--pattern", default="orbit", choices=["orbit", "helix", "flyover", "reveal", "topdown", "capture"])
    s.add_argument("--seconds", type=float, default=20.0); s.add_argument("--fps", type=int, default=30)
    s.add_argument("--size", default="1080p", help="480p, 720p, 1080p, 1440p, 4k or WxH")
    s.add_argument("--elevation", type=float, default=30.0, help="camera elevation angle in degrees")
    s.add_argument("--distance", type=float, default=1.0, help="distance factor (1 = site fills the frame)")
    s.add_argument("--display", default="auto", choices=["auto", "textured", "colours", "shaded", "points"])
    s.add_argument("--start-angle", default="auto", help="degrees, or auto = from the side the photos were taken")
    s.add_argument("--arc", type=float, default=360.0, help="orbit sweep in degrees (e.g. 180 for one-sided scans)")
    s.add_argument("--crf", type=int, default=18); s.add_argument("--out", default=None)
    s = sp("chunks", cmd_chunks)
    s.add_argument("action", choices=["list", "add", "duplicate", "remove", "align", "merge"])
    s.add_argument("idx", nargs="*", type=int); s.add_argument("--label", default=None)
    s.add_argument("--by", default="markers", choices=["markers", "cameras"])
    s.add_argument("--mesh", action="store_true", help="merge: also merge meshes")
    s = sp("export", cmd_export)
    s.add_argument("--model"); s.add_argument("--points"); s.add_argument("--colmap")
    s.add_argument("--sparse", action="store_true"); s.add_argument("--max-faces", type=int, default=None)
    s.add_argument("--markers", help="CSV of markers in world coordinates")
    s = sp("detect-markers", cmd_detect_markers)
    s.add_argument("--bits", type=int, default=12, choices=[12, 14, 16, 20])
    s.add_argument("--min-radius", type=float, default=2.0, help="min centre-dot radius in pixels")
    s.add_argument("--tolerance", type=float, default=0.3)
    s.add_argument("--no-guided", action="store_true"); s.add_argument("--reset", action="store_true")
    s = sp("scalebar", cmd_scalebar); s.add_argument("a"); s.add_argument("b")
    s.add_argument("distance", type=float); s.add_argument("--accuracy", type=float, default=0.001)
    s = sp("reference", cmd_reference); s.add_argument("csv", help="label,x,y,z[,accuracy] per line")
    s = sub.add_parser("gui", help="desktop GUI"); s.add_argument("project", nargs="?")
    s.set_defaults(fn=lambda a: __import__("shapekit.gui", fromlist=["run"]).run(a.project))
    s = sp("report", cmd_report); s.add_argument("--out", default=None)
    s = sub.add_parser("doctor", help="check GPU, models, FFmpeg and the GUI"); s.add_argument("project", nargs="?")
    s.set_defaults(fn=lambda a: sys.exit(__import__("shapekit.tools.doctor", fromlist=["main"]).main(a.project)))
    sp("info", cmd_info)
    s = sp("underwater", cmd_underwater, help="underwater colour correction for the texture (then rebuild the texture)")
    s.add_argument("--mode", choices=["off", "wb", "seathru"], help="wb = white balance from a patch; seathru = "
                   "distance-aware water removal (needs a mesh)")
    s.add_argument("--patch", nargs=5, metavar=("PHOTO", "X0", "Y0", "X1", "Y1"),
                   help="reference patch: photo label/file name and a pixel rectangle in that photo")
    s.add_argument("--no-patch", action="store_true", help="forget the patch (grey-world white point)")
    s.add_argument("--true-color", metavar="R,G,B", help="the patch's real colour in air (default: neutral grey)")
    s.add_argument("--neutral", action="store_true", help="the patch is neutral grey / white")
    s.add_argument("--strength", type=float, default=None, help="distance-aware strength (0-1.5, default 1)")
    s.add_argument("--preview", nargs=2, metavar=("PHOTO", "OUT.jpg"), help="write a before|after preview of one photo")
    s = sub.add_parser("config", help="show or change settings, e.g.  config ffmpeg C:\\ffmpeg\\bin  /  config ffmpeg --clear")
    s.add_argument("key", nargs="?", choices=["ffmpeg"]); s.add_argument("value", nargs="?")
    s.add_argument("--clear", action="store_true", help="remove the saved value")
    s.set_defaults(fn=cmd_config)
    s = sub.add_parser("get-ffmpeg", help="download FFmpeg (Windows) into the app's ffmpeg folder")
    s.add_argument("dest", nargs="?"); s.add_argument("--url", default=None)
    s.set_defaults(fn=lambda a: sys.exit(__import__("shapekit.tools.get_ffmpeg", fromlist=["main"]).main(a.dest, a.url)))
    s = sp("process", cmd_process)
    s.add_argument("--video", nargs="*"); s.add_argument("--photos", nargs="*")
    s.add_argument("--preset", default="balanced", choices=list(PRESETS))
    s.add_argument("--shift", type=float, default=0.08); s.add_argument("--max-gap", type=float, default=1.5)
    s.add_argument("--min-sharpness", type=float, default=0.7,
                   help="skip motion-blurred frames: minimum sharpness relative to nearby frames (0-1, 0 = off)")
    s.add_argument("--workers", type=int, default=None); s.add_argument("--no-cloud", action="store_true")
    s.add_argument("--features", default="auto", choices=["auto", "sift", "superpoint", "aliked", "disk"])
    s.add_argument("--no-markers", action="store_true", help="skip coded-target detection")
    s.add_argument("--depth-method", default="planesweep", choices=["planesweep", "patchmatch"])
    s.add_argument("--blending", default="mosaic", choices=["mosaic", "multiband"])
    s.add_argument("--close-holes", type=int, default=0, help="fill mesh holes up to N edges (0 = off)")
    s.add_argument("--refine", action="store_true", help="photo-consistency mesh refinement")
    s.add_argument("--flythrough", default=None, choices=["orbit", "helix", "flyover", "reveal", "topdown", "capture"],
                   help="also render a fly-through video with this pattern")
    s.add_argument("--resume", dest="resume_from", action="store_const", const="auto", default=None,
                   help="continue a batch run that crashed: skip the steps it had finished")
    a = ap.parse_args(argv)
    proj = getattr(a, "project", None)
    if not proj or a.cmd == "gui" or not os.path.isdir(proj):
        return a.fn(a)
    from . import projlog                       # append this run to <project>/shapekit.log
    crashed = projlog.crashed_task(proj)
    projlog.open_log(proj, "cli " + " ".join(sys.argv[1:] if argv is None else argv))
    if crashed:
        msg = "the previous run crashed during '%s'%s (started %s, ran %s)" % (
            crashed.get("title"), ", step '%s'" % crashed["stage"] if crashed.get("stage") else "",
            crashed.get("started"), crashed.get("ran_for"))
        projlog.write("!!!!! " + msg)
        hint = ("  -> add --resume to continue from that step"
                if a.cmd == "process" and not getattr(a, "resume_from", None) else "")
        print("NOTE: %s%s" % (msg, ("\n" + hint) if hint else ""), flush=True)
    if getattr(a, "resume_from", None) == "auto":
        a.resume_from = (crashed or {}).get("stage")
        if not a.resume_from:
            print("NOTE: --resume: no crashed batch step recorded - running all steps", flush=True)
    title = "cli " + a.cmd
    projlog.task_start(title)
    try:
        r = a.fn(a)
    except SystemExit as e:
        projlog.task_end(title, None if e.code in (0, None) else "exit code %s" % e.code)
        projlog.close_log()
        raise
    except BaseException:
        import traceback
        projlog.task_end(title, traceback.format_exc())
        projlog.close_log("error")
        raise
    projlog.task_end(title)
    projlog.close_log()
    return r


if __name__ == "__main__":
    main()
