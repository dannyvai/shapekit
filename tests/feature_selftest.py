"""Self-test for GPU rasterisation / texture visibility and learned retrieval (SALAD vs VLAD).

uv run python -m tests.feature_selftest projects\\gpu_test projects\\area_d
"""
import os
import sys
import time
import numpy as np


def raster_test(proj_dir):
    from shapekit.project import Project
    from shapekit import raster, camera as cm
    from shapekit.mesh import load_mesh
    ch = Project.open(proj_dir).chunk()
    V, F, _ = load_mesh(ch)
    print("\n== GPU vs CPU rasteriser on %s (%d faces)" % (proj_dir, len(F)))
    al = ch.aligned_cameras()
    for i, scale in [(al[len(al) // 3], 0.5), (al[2 * len(al) // 3], 1.0)]:
        cam = ch.cameras[i]; sen = ch.sensors[cam.sensor]
        K = cm.K_matrix(sen.intr, scale); w, h = int(sen.width * scale), int(sen.height * scale)
        uv, z = raster.project_vertices(V, cam.rvec, cam.tvec, K)
        raster.rasterize(uv, z, F, w, h, gpu=True)            # warm-up
        t = time.time(); ig, zg = raster.rasterize(uv, z, F, w, h, gpu=True); tg = time.time() - t
        t = time.time(); ic, zc = raster.rasterize(uv, z, F, w, h, gpu=False); tc = time.time() - t
        both = (ig >= 0) & (ic >= 0)
        same = (ig == ic)[both].mean()
        zerr = np.abs(zg[both] - zc[both]).max() if both.any() else 0
        cov = ((ig >= 0) != (ic >= 0)).mean()
        print("  %dx%d: GPU %.3fs, CPU %.2fs (x%.0f)  same face id %.2f%%, coverage mismatch %.3f%%, max depth diff %.2e"
              % (w, h, tg, tc, tc / max(tg, 1e-6), 100 * same, 100 * cov, zerr))
        assert same > 0.995 and cov < 0.002, "GPU raster disagrees with CPU"


def texture_test(proj_dir):
    import shutil
    from shapekit.project import Project
    from shapekit.texture import build_texture
    print("\n== Build Texture timing on a copy of %s" % proj_dir)
    dst = proj_dir.rstrip("\\/") + "_textest"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(proj_dir, dst, ignore=shutil.ignore_patterns("depth", "features", "exports", "*.html",
                                                                  "point_cloud.ply"))
    p = Project.open(dst); ch = p.chunk()
    t = time.time(); build_texture(ch, 8192); dt = time.time() - t
    print("  Build Texture total: %.1fs (previous CPU run on this mesh: ~100s, ~90s of it visibility)" % dt)
    shutil.rmtree(dst, ignore_errors=True)


def retrieval_test(proj_dir, n_eval=40, topk=5, seed=0):
    """Fair benchmark: candidates from each descriptor are verified with ALIKED + LightGlue + MAGSAC at full
    resolution (independent of every descriptor under test). Metric: per query photo, the top-k candidates
    from OTHER videos; a candidate is a verified loop closure if it has >= 50 inliers."""
    import cv2
    from shapekit.project import Project
    from shapekit import features as Fm
    from shapekit.util import read_image
    from shapekit.learned import get as get_lf
    from shapekit.learned.retrieval import global_descriptors
    ch = Project.open(proj_dir).chunk()
    print("\n== Retrieval on %s: %d photos from %d videos" % (proj_dir, len(ch.cameras),
                                                             len({c.source for c in ch.cameras})))
    descs = {}
    for kind in ("salad", "dinov2"):
        t = time.time(); descs[kind] = global_descriptors(ch, kind); print("  %s descriptors: %.1fs" % (kind, time.time() - t))
    t = time.time(); descs["vlad"] = Fm.vlad_descriptors(ch)
    print("  vlad descriptors (from project %s features): %.1fs" % (Fm.feature_method(ch), time.time() - t))
    src = np.array([c.source for c in ch.cameras])
    rng = np.random.default_rng(seed)
    q = rng.choice(len(ch.cameras), min(n_eval, len(ch.cameras)), replace=False)

    def ranked(D, a):
        S = D @ D[a]
        cand = np.where(src != src[a])[0]
        return list(cand[np.argsort(-S[cand])])
    cands = {}
    for a in q:
        r = {k: ranked(D, a) for k, D in descs.items()}
        hyb = []
        for x in [v for pair in zip(r["salad"], r["vlad"]) for v in pair]:
            if x not in hyb:
                hyb.append(x)
            if len(hyb) >= topk:
                break
        r = {k: v[:topk] for k, v in r.items()}
        r["hybrid"] = hyb
        cands[a] = r
    lf = get_lf("aliked", 4096, None)
    feats = {}

    def F(i):
        if i not in feats:
            feats[i] = lf.extract(read_image(ch.abspath(ch.cameras[i])))
        return feats[i]
    cache = {}

    def verified(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            fa, fb = F(a), F(b)
            sz = (ch.sensors[0].width, ch.sensors[0].height)
            m, _ = lf.match((fa["xy"], fa["desc"]), (fb["xy"], fb["desc"]), sz, sz)
            n = 0
            if len(m) >= 15:
                _, mask = cv2.findFundamentalMat(fa["xy"][m[:, 0]], fb["xy"][m[:, 1]], cv2.USAC_MAGSAC, 1.2, 0.9999, 10000)
                n = int(mask.sum()) if mask is not None else 0
            cache[key] = n
        return cache[key]
    t = time.time()
    print("  %-7s %s" % ("method", "queries with >=1 verified loop closure in top-%d | verified pairs / candidates | median inliers of best" % topk))
    for k in ("salad", "dinov2", "vlad", "hybrid"):
        hit, ver, tot, best = 0, 0, 0, []
        for a in q:
            ns = [verified(a, b) for b in cands[a][k]]
            hit += any(n >= 50 for n in ns); ver += sum(n >= 50 for n in ns); tot += len(ns); best.append(max(ns))
        print("  %-7s %3d/%d                                         | %3d/%d | %d" % (k, hit, len(q), ver, tot, int(np.median(best))))
    print("  (verification of %d pairs took %.0fs)" % (len(cache), time.time() - t))


def main(argv):
    gpu_proj = argv[0] if argv else "projects/gpu_test"
    multi = argv[1] if len(argv) > 1 else None
    import torch
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    raster_test(gpu_proj)
    texture_test(gpu_proj)
    if multi and os.path.exists(os.path.join(multi, "project.json")):
        retrieval_test(multi)
    print("\nFEATURE SELFTEST OK")


if __name__ == "__main__":
    main(sys.argv[1:])
