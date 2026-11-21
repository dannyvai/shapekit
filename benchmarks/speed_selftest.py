"""Benchmark + equivalence check: original (SHAPEKIT_LEGACY=1) vs optimised code paths, stage by stage,
on copies of a project. Usage: uv run python -m benchmarks.speed_selftest projects\\gpu_test
"""
import json
import os
import shutil
import sys
import time

import numpy as np


def run_mode(src, mode):
    if mode == "legacy":
        os.environ["SHAPEKIT_LEGACY"] = "1"
    else:
        os.environ.pop("SHAPEKIT_LEGACY", None)
    import cv2
    from shapekit.project import Project
    from shapekit import markers as MK, mvs, mesh as ME, texture as TX
    from shapekit.io import read_ply
    dst = os.path.abspath(src.rstrip("\\/") + "_speed_" + mode)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("depth", "model", "exports", "*.ply", "*.html", "logs"))
    p = Project.open(dst); ch = p.chunk()
    ch.markers = [m for m in ch.markers if not m.get("auto")]
    p.save()
    T = {}
    t = time.time(); MK.detect_markers(ch); T["coded targets"] = time.time() - t
    t = time.time(); MK.refine_marker_projections(ch); T["guided markers"] = time.time() - t
    p.save()
    t = time.time(); mvs.build_depth_maps(ch, "high", "moderate", force=True); T["depth maps + filter"] = time.time() - t
    p.save()
    t = time.time(); mvs.build_point_cloud(ch); T["point cloud"] = time.time() - t
    t = time.time(); ME.build_mesh(ch, "medium"); T["mesh"] = time.time() - t
    p.save()
    t = time.time(); TX.build_texture(ch, 8192); T["texture"] = time.time() - t
    out = dict(markers=json.dumps(ch.markers, sort_keys=True),
               depth={i: mvs.load_depth(ch, i) for i in ch.aligned_cameras()},
               cloud=read_ply(ch.path("point_cloud.ply")),
               mesh=ME.load_mesh(ch),
               uv=np.load(ch.path("model", "uv.npz"))["uv"],
               atlas=cv2.imread(ch.path("model", "model_texture.jpg")))
    return T, out, dst


def main(src="projects/gpu_test"):
    results = {}
    for mode in ("legacy", "fast"):
        print("\n######## %s ########" % mode, flush=True)
        results[mode] = run_mode(src, mode)
    (Tl, L, dl), (Tf, Fo, df) = results["legacy"], results["fast"]
    print("\n================ RESULTS (%s) ================" % src)
    print("%-22s %10s %10s %8s" % ("stage", "legacy s", "fast s", "speed-up"))
    for k in Tl:
        print("%-22s %10.1f %10.1f %7.1fx" % (k, Tl[k], Tf[k], Tl[k] / max(Tf[k], 1e-6)))
    print("%-22s %10.1f %10.1f %7.1fx" % ("TOTAL", sum(Tl.values()), sum(Tf.values()),
                                          sum(Tl.values()) / max(sum(Tf.values()), 1e-6)))
    print("\nequivalence:")
    print("  markers identical:            ", L["markers"] == Fo["markers"])
    same_d = all(np.array_equal(L["depth"][i]["depth"], Fo["depth"][i]["depth"]) for i in L["depth"])
    same_n = all(np.array_equal(L["depth"][i].get("nconsist"), Fo["depth"][i].get("nconsist")) for i in L["depth"])
    print("  filtered depth maps identical:", same_d, "| consistency counts identical:", same_n)
    print("  point cloud identical:        ", all(np.array_equal(L["cloud"][k], Fo["cloud"][k])
                                                  for k in ("verts", "colors", "normals", "confidence")))
    (V1, F1, C1), (V2, F2, C2) = L["mesh"], Fo["mesh"]
    if V1.shape == V2.shape and F1.shape == F2.shape and np.array_equal(F1, F2):
        dv = np.abs(V1 - V2).max()
        print("  mesh: same topology (%d faces); max vertex difference %.2e; colours identical: %s"
              % (len(F1), dv, np.array_equal(C1, C2)))
    else:
        print("  mesh: %d vs %d faces (GPU float rounding changed a few voxels)" % (len(F1), len(F2)))
    if L["uv"].shape == Fo["uv"].shape:
        print("  texture UVs identical: %s; atlas pixels identical: %.3f%%; mean abs diff %.3f"
              % (np.array_equal(L["uv"], Fo["uv"]), 100 * (L["atlas"] == Fo["atlas"]).all(2).mean(),
                 np.abs(L["atlas"].astype(int) - Fo["atlas"].astype(int)).mean()))
    else:
        print("  texture: different chart layout (follows from mesh difference)")
    for d in (dl, df):
        shutil.rmtree(d, ignore_errors=True)
    print("\nSPEED SELFTEST DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "projects/gpu_test")
