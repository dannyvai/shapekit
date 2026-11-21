"""Self-test for the roadmap features on the user's machine (GPU).

uv run python -m tests.roadmap_selftest projects\\gpu_test

Works on copies (projects\\_roadmap_*), never on the source project.
A  GPU bundle adjustment: mixed vs double precision; fisheye model CPU vs GPU
B  PatchMatch depth: synthetic slanted scene with ground truth (GPU vs plane sweep), real views (speed/agreement)
C  Fisheye end-to-end: the source photos re-rendered through an equidistant fisheye lens, aligned with ALIKED
D  Masks: masked pixels get no depth; AI masks on a few photos
E  Mesh tools: close holes, refine mesh; texture multiband vs mosaic (seam colour step)
F  Chunks: duplicate, align by cameras, merge
G  Full PatchMatch depth maps -> point cloud -> mesh on the project copy
"""
import os
import sys
import time
import glob
import shutil
import traceback
import numpy as np
import cv2

RESULTS = []


def say(*a):
    print(" ".join(str(x) for x in a), flush=True)


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok), info))
    say("  [%s] %s  %s" % ("PASS" if ok else "FAIL", name, info))


def section(title, fn):
    say("\n=== %s" % title)
    t = time.time()
    try:
        fn()
    except Exception:
        RESULTS.append((title, False, "exception"))
        say("  [FAIL] %s raised:\n%s" % (title, traceback.format_exc()))
    say("  (%.1f s)" % (time.time() - t))


def copy_project(src, name):
    dst = os.path.join(os.path.dirname(os.path.abspath(src)), name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("exports", "*.html", "*_guitest*"))
    return dst


# ----------------------------------------------------------------------------- A
def test_ba():
    from tests.ba_selftest import scene
    from shapekit.learned.ba_gpu import bundle_adjust_gpu
    from shapekit.ba import bundle_adjust
    from shapekit import camera as cm
    cr, ct, P, intr, oc, op, uv, rng = scene(300, 30000)
    cr0, ct0, P0 = cr + rng.normal(0, 0.003, cr.shape), ct + rng.normal(0, 0.02, ct.shape), P + rng.normal(0, 0.02, P.shape)
    i0 = intr.copy(); i0[0, 0] = 1450; i0[0, 3] = 0
    fixed = np.zeros(300, bool); fixed[0] = True
    mask = np.array([1, 1, 1, 1, 1, 0, 0, 0], bool)
    out = {}
    for prec in ("double", "mixed", "double"):            # double twice: first run includes CUDA warm-up
        t = time.time()
        r, tt, X, I, info = bundle_adjust_gpu(cr0, ct0, P0, i0, np.zeros(300, int), oc, op, uv, fixed_cams=fixed,
                                              intr_mask=mask, iters=15, huber=2.0, precision=prec)
        out[prec] = (time.time() - t, info["final_rms"], I[0, 0], info["iterations"])
    for prec, (dt, rms, f, it) in out.items():
        say("  %-6s %.2fs, %d iterations, rms %.4f px, f %.2f (true 1500)" % (prec, dt, it, rms, f))
    d, m = out["double"], out["mixed"]
    check("mixed-precision BA converges like double", abs(m[1] - d[1]) < 0.01 * d[1] + 1e-3 and abs(m[2] - 1500) < 3,
          "rms %.4f vs %.4f, speed-up %.1fx" % (m[1], d[1], d[0] / max(m[0], 1e-6)))
    # fisheye: GPU vs CPU
    cr, ct, P, intr, oc, op, uv, rng = scene(40, 4000, seed=3)
    fe = np.array([[900, 960, 540, 0.01, 0, 0, 0, 0, 1.0]])
    uv, z = cm.project(cr[oc], ct[oc], P[op], fe[np.zeros(len(oc), int)])
    ok = (z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
    oc, op, uv = oc[ok], op[ok], uv[ok] + rng.normal(0, 0.3, (ok.sum(), 2))
    f0 = fe.copy(); f0[0, 0] = 860; f0[0, 3] = 0
    fixed = np.zeros(40, bool); fixed[0] = True
    m5 = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    cr0, ct0, P0 = cr + rng.normal(0, 0.002, cr.shape), ct + rng.normal(0, 0.01, ct.shape), P + rng.normal(0, 0.01, P.shape)
    res = {}
    for name, fn in (("GPU", bundle_adjust_gpu), ("CPU", bundle_adjust)):
        r, tt, X, I, info = fn(cr0, ct0, P0, f0, np.zeros(40, int), oc, op, uv, fixed_cams=fixed, intr_mask=m5, iters=20)
        res[name] = (I[0, 0], info["final_rms"], X)
        say("  fisheye %s: f %.2f (true 900), rms %.3f" % (name, I[0, 0], info["final_rms"]))
    check("fisheye BA: GPU == CPU and f recovered", abs(res["GPU"][0] - res["CPU"][0]) < 0.05 and abs(res["GPU"][0] - 900) < 3,
          "f GPU %.3f CPU %.3f" % (res["GPU"][0], res["CPU"][0]))


# ----------------------------------------------------------------------------- B
def test_patchmatch(src):
    from shapekit.learned.synthetic import synthetic_views, _shrink
    from shapekit.learned.patchmatch import patchmatch_depth
    from shapekit.learned.mvs_gpu import compute_depth_gpu
    from shapekit.project import Project
    from shapekit import mvs
    ch = Project.open(src).chunk()
    tex = ch.abspath(ch.cameras[len(ch.cameras) // 2])
    views = synthetic_views(tex)
    ref, nbrs = views[2], [views[j] for j in (0, 1, 3, 4, 5)]
    gt = ref.gt; lo, hi = 0.75 * gt[gt > 0].min(), 1.3 * gt[gt > 0].max()
    out = {}
    patchmatch_depth(ref, nbrs, lo, hi)                    # warm-up
    for name, fn in (("plane sweep", lambda: compute_depth_gpu(ref, nbrs, lo, hi, 192, 9)),
                     ("PatchMatch", lambda: patchmatch_depth(ref, nbrs, lo, hi))):
        t = time.time(); d, _ = fn(); dt = time.time() - t
        m = (d > 0) & (gt > 0); rel = np.abs(d[m] - gt[m]) / gt[m]
        out[name] = (np.median(rel), (rel < 0.01).mean(), m.sum() / (gt > 0).sum(), dt)
        say("  synthetic %-11s %.2fs: valid %.0f%%, median rel err %.4f, within 1%%: %.0f%%" %
            (name, dt, 100 * out[name][2], out[name][0], 100 * out[name][1]))
    check("PatchMatch more accurate than plane sweep on slanted ground (ground truth)",
          out["PatchMatch"][0] < out["plane sweep"][0] and out["PatchMatch"][2] > 0.85,
          "median rel err %.4f vs %.4f" % (out["PatchMatch"][0], out["plane sweep"][0]))
    # numpy reference implementation gives the same quality as the torch one (same algorithm)
    small = [_shrink(v, 0.25) for v in views]          # NumPy reference is CPU-only: keep it small
    r2, n2 = small[2], [small[j] for j in (0, 1, 3, 4, 5)]
    dn, _ = patchmatch_depth(r2, n2, lo, hi, prefer="numpy")
    dg, _ = patchmatch_depth(r2, n2, lo, hi, prefer="auto")
    g2 = r2.gt
    e = lambda d: np.median(np.abs(d[(d > 0) & (g2 > 0)] - g2[(d > 0) & (g2 > 0)]) / g2[(d > 0) & (g2 > 0)])
    check("PatchMatch GPU backend matches the NumPy reference quality", e(dg) < 1.3 * e(dn) + 1e-3,
          "median rel err GPU %.4f, NumPy %.4f" % (e(dg), e(dn)))
    # real views
    nb = mvs.select_neighbors(ch, 5)
    cams = [i for i in ch.aligned_cameras() if nb.get(i)][::max(1, len(ch.aligned_cameras()) // 4)][:4]
    tp, tps, agree = [], [], []
    for i in cams:
        refv = mvs.View(ch, i, 0.5); nbv = [mvs.View(ch, j, 0.5) for j in nb[i]]
        z0, z1 = mvs.depth_range(ch, i, refv)
        t = time.time(); dps, _ = compute_depth_gpu(refv, nbv, z0, z1, 192, 9); tps.append(time.time() - t)
        t = time.time(); dpm, _ = patchmatch_depth(refv, nbv, z0, z1); tp.append(time.time() - t)
        m = (dps > 0) & (dpm > 0)
        agree.append((np.abs(dps[m] - dpm[m]) / dps[m] < 0.01).mean())
        say("  view %d (%dx%d): plane sweep %.2fs %.0f%% valid | PatchMatch %.2fs %.0f%% valid | %.0f%% agree within 1%%"
            % (i, refv.w, refv.h, tps[-1], 100 * (dps > 0).mean(), tp[-1], 100 * (dpm > 0).mean(), 100 * agree[-1]))
    check("PatchMatch on real views agrees with plane sweep", np.mean(agree) > 0.6,
          "mean agreement %.0f%%, %.2f s/view (plane sweep %.2f s/view)" % (100 * np.mean(agree), np.mean(tp), np.mean(tps)))


# ----------------------------------------------------------------------------- C
def test_fisheye(src):
    from shapekit.project import Project
    from shapekit import camera as cm
    from shapekit.video import add_photos
    from shapekit.sfm import align_photos
    ch0 = Project.open(src).chunk()
    sen0 = ch0.sensors[ch0.cameras[0].sensor]
    I0 = sen0.calib()
    W, H = sen0.width, sen0.height
    th_edge = np.arctan((W - I0[1]) / I0[0])
    fe = (W / 2) / th_edge
    Ife = np.array([fe, W / 2 - 0.5, H / 2 - 0.5, 0, 0, 0, 0, 0, 1.0])
    base = os.path.join(os.path.dirname(os.path.abspath(src)), "_roadmap_fisheye")
    if os.path.exists(base):
        shutil.rmtree(base)
    os.makedirs(os.path.join(base, "in"))
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    xn = cm.undistort_points(np.stack([xs.ravel(), ys.ravel()], 1), Ife)
    mp = cm.distort(xn, I0).reshape(H, W, 2).astype(np.float32)
    cams = [c for c in ch0.cameras if c.sensor == ch0.cameras[0].sensor]
    for c in cams:
        img = cv2.imread(ch0.abspath(c))
        cv2.imwrite(os.path.join(base, "in", c.label + ".jpg"),
                    cv2.remap(img, mp[..., 0], mp[..., 1], cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    p = Project.create(os.path.join(base, "proj")); ch = p.chunk()
    add_photos(ch, sorted(glob.glob(os.path.join(base, "in", "*.jpg"))))
    ch.sensors[0].set_model("fisheye_equidistant"); ch.sensors[0].intr[0] = fe * 0.9
    p.save()
    from shapekit.cli import resolve_features
    st = align_photos(ch, "high", 4096, 12, 10, method=resolve_features("auto"), use_markers=False)
    p.save()
    f = ch.sensors[0].intr[0]
    say("  fisheye: %d/%d aligned, rms %.3f px, f %.1f (true %.1f, start %.1f)" % (st["aligned"], st["cameras"],
                                                                                st["rms_reprojection_px"], f, fe, fe * 0.9))
    check("fisheye photos align and f is recovered", st["aligned"] >= 0.9 * st["cameras"] and abs(f - fe) < 0.01 * fe,
          "f error %.2f%%" % (100 * abs(f - fe) / fe))


# ----------------------------------------------------------------------------- D
def test_masks(dst):
    from shapekit.project import Project
    from shapekit import masks as MS, mvs
    p = Project.open(dst); ch = p.chunk()
    i = ch.aligned_cameras()[3]
    sen = ch.sensors[ch.cameras[i].sensor]
    keep = np.ones((sen.height, sen.width), bool); keep[:, : sen.width // 2] = False
    MS.save_mask(ch, i, keep)
    v = mvs.View(ch, i, 0.5)
    check("masked half of a photo is excluded from depth", not v.valid[:, : v.w // 2 - 4].any() and v.valid[:, v.w // 2 + 4:].mean() > 0.9)
    MS.clear_mask(ch, i)
    cams = ch.aligned_cameras()[::max(1, len(ch.aligned_cameras()) // 6)][:6]
    t = time.time()
    n = MS.generate_ai_masks(ch, ("people", "vehicles"), 12, cameras=cams)
    dt = time.time() - t
    check("AI masks run", True, "%d of %d photos had people/vehicles, %.2f s/photo" % (n, len(cams), dt / len(cams)))
    for i in MS.masked_cameras(ch):
        MS.clear_mask(ch, i)


# ----------------------------------------------------------------------------- E
def test_mesh_texture(dst):
    from shapekit.project import Project
    from shapekit import mesh as M
    from shapekit.texture import build_texture, face_adjacency
    p = Project.open(dst); ch = p.chunk()
    if not ch.stats.get("mesh"):
        say("  no mesh in the project, skipped"); return
    V, F, _ = M.load_mesh(ch)
    t = time.time(); n = M.close_holes_chunk(ch, 200); dt = time.time() - t
    V2, F2, _ = M.load_mesh(ch)
    e = np.concatenate([F2[:, [0, 1]], F2[:, [1, 2]], F2[:, [2, 0]]]).astype(np.int64)
    k = e[:, 0] * len(V2) + e[:, 1]
    check("close holes keeps the mesh consistently oriented", len(k) == len(np.unique(k)),
          "%d holes filled, +%d faces, %.1f s" % (n, len(F2) - len(F), dt))
    t = time.time(); M.refine_mesh(ch, iterations=2); dt = time.time() - t
    check("refine mesh runs", True, "%d vertices, %.1f s for 2 iterations" % (len(V2), dt))
    p.save()
    steps = {}
    for mode in ("mosaic", "multiband"):
        t = time.time(); build_texture(ch, 8192, blending=mode); dt = time.time() - t
        steps[mode] = (_seam_step(ch), dt)
        say("  texture %-9s %.1f s, mean colour step across chart seams %.1f levels" % (mode, dt, steps[mode][0]))
    check("multiband blending reduces seam colour steps", steps["multiband"][0] < steps["mosaic"][0],
          "%.1f -> %.1f levels" % (steps["mosaic"][0], steps["multiband"][0]))
    p.save()


def _seam_step(ch, n=20000):
    from shapekit.mesh import load_mesh
    from shapekit.texture import face_adjacency
    V, F, _ = load_mesh(ch)
    atlas = cv2.imread(ch.path("model", "model_texture.jpg")).astype(float)
    uv = np.load(ch.path("model", "uv.npz")); UV, FU = uv["uv"], uv["face_uv"]
    a, b = face_adjacency(F)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(a), min(n, len(a)), replace=False)
    S = atlas.shape[0]
    diffs = []
    for x, y in zip(a[sel], b[sel]):
        sh = list(set(F[x]) & set(F[y]))
        if len(sh) != 2:
            continue
        ia = [list(F[x]).index(v) for v in sh]; ib = [list(F[y]).index(v) for v in sh]
        ta = 0.5 * (UV[FU[x, ia[0]]] + UV[FU[x, ia[1]]]); tb = 0.5 * (UV[FU[y, ib[0]]] + UV[FU[y, ib[1]]])
        if np.abs(ta - tb).max() < 1e-6:
            continue                                        # same chart: no seam
        ca = atlas[min(S - 1, int((1 - ta[1]) * S)), min(S - 1, int(ta[0] * S))]
        cb = atlas[min(S - 1, int((1 - tb[1]) * S)), min(S - 1, int(tb[0] * S))]
        diffs.append(np.abs(ca - cb).mean())
    return float(np.mean(diffs)) if diffs else 0.0


# ----------------------------------------------------------------------------- F
def test_chunks(dst):
    from shapekit.project import Project
    from shapekit import chunks as CK
    p = Project.open(dst)
    n0 = len(p.chunks[0].cameras)
    CK.duplicate_chunk(p, 0, "copy", items=("alignment",))
    r = CK.align_chunks(p, 0, [1], "cameras")
    m = CK.merge_chunks(p, [0, 1], "merged", merge_depth=False, merge_cloud=False)
    p.save()
    check("duplicate / align / merge chunks", r[1]["rmse"] < 1e-6 and len(m.cameras) == 2 * n0,
          "align RMSE %.2e, merged %d cameras" % (r[1]["rmse"], len(m.cameras)))
    p = Project.open(dst)
    for _ in range(len(p.chunks) - 1):
        CK.remove_chunk(p, 1)
    p.save()


# ----------------------------------------------------------------------------- G
def test_full_patchmatch(dst):
    from shapekit.project import Project
    from shapekit.mvs import build_depth_maps, build_point_cloud
    from shapekit.mesh import build_mesh
    p = Project.open(dst); ch = p.chunk()
    t = time.time()
    build_depth_maps(ch, "high", "moderate", force=True, method="patchmatch")
    dt = time.time() - t
    n = len(ch.aligned_cameras())
    build_point_cloud(ch); build_mesh(ch, "medium"); p.save()
    check("PatchMatch depth maps -> point cloud -> mesh", ch.stats["mesh"]["faces"] > 1000,
          "%.2f s/view incl. filtering, %d points, %d faces" % (dt / n, ch.stats["point_cloud"]["points"], ch.stats["mesh"]["faces"]))


def main(src="projects/gpu_test"):
    import torch
    say("ROADMAP SELFTEST on %s | torch %s | CUDA %s %s" % (src, torch.__version__, torch.cuda.is_available(),
                                                          torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))
    dst = copy_project(src, "_roadmap_test")
    section("A  GPU bundle adjustment: mixed precision, fisheye", test_ba)
    section("B  PatchMatch depth", lambda: test_patchmatch(dst))
    section("C  Fisheye end-to-end alignment", lambda: test_fisheye(src))
    section("D  Masks", lambda: test_masks(dst))
    section("E  Mesh tools + multiband texture", lambda: test_mesh_texture(dst))
    section("F  Chunks", lambda: test_chunks(dst))
    section("G  PatchMatch depth -> cloud -> mesh", lambda: test_full_patchmatch(dst))
    np_ = sum(ok for _, ok, _ in RESULTS)
    say("\nSUMMARY")
    for name, ok, info in RESULTS:
        say("  %s  %-60s %s" % ("PASS" if ok else "FAIL", name, info))
    say("ROADMAP SELFTEST: %d passed, %d failed" % (np_, len(RESULTS) - np_))
    return 0 if np_ == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
