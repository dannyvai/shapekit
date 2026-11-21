"""Build Depth Maps + Build Point Cloud.

Depth maps: multi-view plane sweep with windowed NCC (occlusion-robust best-k aggregation over
neighbour views), sub-plane refinement, texture/confidence masking.
Filtering: geometric consistency across neighbours (Metashape's mild / moderate / aggressive).
Point cloud: back-projection of consistent depths, coloured, with normals and confidence, voxel-merged.
"""
import os
import time
import numpy as np
import cv2

from . import camera as cm
from .util import log, read_image

QUALITY = {"ultra": 1.0, "high": 0.5, "medium": 0.25, "low": 0.125, "lowest": 0.0625}
FILTER = {"none": 0, "mild": 1, "moderate": 2, "aggressive": 3}


class View:
    """Undistorted, rescaled pinhole view of an aligned camera."""

    def __init__(self, chunk, i, scale):
        cam = chunk.cameras[i]
        sen = chunk.sensors[cam.sensor]
        img = read_image(chunk.abspath(cam))
        m1, m2, self.K = cm.undistort_maps(sen.calib(), sen.width, sen.height, scale)
        if scale < 1:
            # pre-blur to avoid aliasing, then remap
            img = cv2.GaussianBlur(img, (0, 0), 0.5 / scale * 0.8)
        self.color = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.valid = cv2.remap(np.ones(img.shape[:2], np.uint8), m1, m2, cv2.INTER_NEAREST) > 0
        from .masks import load_mask
        mk = load_mask(chunk, i, (img.shape[1], img.shape[0]))
        if mk is not None:                         # masked pixels get no depth
            self.valid &= cv2.remap(mk.astype(np.uint8), m1, m2, cv2.INTER_NEAREST) > 0
        self.R = cm.rodrigues_to_R(cam.rvec)[0]
        self.t = np.asarray(cam.tvec, float)
        self.C = -self.R.T @ self.t
        self.h, self.w = self.gray.shape
        self.Kinv = np.linalg.inv(self.K)

    def backproject(self, depth):
        """depth (h,w) -> world points (h,w,3)."""
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        rays = np.stack([xs, ys, np.ones_like(xs)], -1) @ self.Kinv.T
        Xc = rays * depth[..., None]
        return (Xc - self.t) @ self.R  # R^T (Xc - t)


def select_neighbors(chunk, n_neighbors=5, theta0=8.0):
    sp = chunk.load_sparse()
    X, oc, op = sp["points"], sp["obs_cam"], sp["obs_pt"]
    aligned = chunk.aligned_cameras()
    C = {i: chunk.cameras[i].center for i in aligned}
    nb = {}
    for i in aligned:
        pts_i = op[oc == i]
        if len(pts_i) < 10:
            nb[i] = []
            continue
        mask_pts = np.zeros(len(X), bool); mask_pts[pts_i] = True
        sel = mask_pts[op] & (oc != i)
        cams, shared = np.unique(oc[sel], return_counts=True)
        P = np.median(X[pts_i], axis=0)
        scores = []
        for c, s in zip(cams, shared):
            if c not in C or s < 20:
                continue
            a = P - C[i]; b = P - C[c]
            th = np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
            if th < 1.0 or th > 60:
                continue
            sig = 4.0 if th < theta0 else 15.0
            scores.append((s * np.exp(-(th - theta0) ** 2 / (2 * sig ** 2)), int(c)))
        scores.sort(reverse=True)
        nb[i] = [c for _, c in scores[:n_neighbors]]
    return nb


def depth_range(chunk, i, view):
    sp = chunk.load_sparse() if not hasattr(chunk, "_sp_cache") else chunk._sp_cache
    chunk._sp_cache = sp
    pts = sp["points"][sp["obs_pt"][sp["obs_cam"] == i]]
    z = (pts @ view.R.T + view.t)[:, 2]
    z = z[z > 0]
    if len(z) < 10:
        return None
    lo, hi = np.percentile(z, 1), np.percentile(z, 99)
    return 0.75 * lo, 1.3 * hi


def _ncc_prep(img, win):
    mu = cv2.blur(img, (win, win))
    var = cv2.blur(img * img, (win, win)) - mu * mu
    return mu, np.maximum(var, 1e-6)


def compute_depth(ref, nbrs, zmin, zmax, n_planes=128, win=7, best_k=2):
    h, w = ref.h, ref.w
    mu_r, var_r = _ncc_prep(ref.gray, win)
    inv = np.linspace(1 / zmin, 1 / zmax, n_planes)
    depths = 1 / inv
    cost = np.empty((n_planes, h, w), np.float16)
    Hs = []
    for nb in nbrs:
        R_rel = nb.R @ ref.R.T
        t_rel = nb.t - R_rel @ ref.t
        Hs.append((nb, R_rel, t_rel))
    n = np.array([0, 0, 1.0])
    ones = {}            # validity image per neighbour size (videos may differ in orientation/resolution)
    per = np.empty((len(nbrs), h, w), np.float32)
    for d_i, d in enumerate(depths):
        for k, (nb, R_rel, t_rel) in enumerate(Hs):
            H = nb.K @ (R_rel + np.outer(t_rel, n) / d) @ ref.Kinv
            wimg = cv2.warpPerspective(nb.gray, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            one = ones.setdefault((nb.h, nb.w), np.ones((nb.h, nb.w), np.float32))
            wval = cv2.warpPerspective(one, H, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mu_n = cv2.blur(wimg, (win, win))
            var_n = np.maximum(cv2.blur(wimg * wimg, (win, win)) - mu_n * mu_n, 1e-6)
            cov = cv2.blur(ref.gray * wimg, (win, win)) - mu_r * mu_n
            ncc = cov / np.sqrt(var_r * var_n)
            c = 1 - ncc
            c[wval < 0.5] = 2.0
            per[k] = c
        if len(nbrs) > best_k:
            part = np.partition(per, best_k - 1, axis=0)[:best_k]
            cost[d_i] = part.mean(0)
        else:
            cost[d_i] = per.mean(0)
    best = np.argmin(cost, axis=0)
    cmin = np.take_along_axis(cost, best[None], 0)[0].astype(np.float32)
    # sub-plane parabola refinement in inverse depth
    b0 = np.clip(best, 1, n_planes - 2)
    c0 = np.take_along_axis(cost, (b0 - 1)[None], 0)[0].astype(np.float32)
    c1 = np.take_along_axis(cost, b0[None], 0)[0].astype(np.float32)
    c2 = np.take_along_axis(cost, (b0 + 1)[None], 0)[0].astype(np.float32)
    denom = c0 - 2 * c1 + c2
    off = np.where(denom > 1e-6, 0.5 * (c0 - c2) / np.maximum(denom, 1e-6), 0)
    off = np.clip(off, -0.5, 0.5)
    step = inv[1] - inv[0]
    inv_d = inv[b0] + off * step
    depth = 1 / inv_d
    edge = (best == 0) | (best == n_planes - 1)
    tex = np.sqrt(var_r) > 0.012
    mask = (cmin < 0.6) & ~edge & tex & ref.valid
    depth[~mask] = 0
    return depth.astype(np.float32), (1 - cmin).astype(np.float32)


def build_depth_maps(chunk, quality="medium", filter_mode="moderate", n_neighbors=5, n_planes=None, force=False,
                     workers=None, gpu=None, method="planesweep"):
    """method: 'planesweep' (fronto-parallel sweep, default) or 'patchmatch' (slanted planes with normals)."""
    t0 = time.time()
    prev_filter = (chunk.stats.get("depth_maps") or {}).get("filter")
    scale = QUALITY[quality]
    nbmap = select_neighbors(chunk, n_neighbors)
    aligned = chunk.aligned_cameras()
    n_planes = n_planes or {1.0: 256, 0.5: 192, 0.25: 128, 0.125: 96, 0.0625: 64}[scale]
    win = 9 if scale >= 0.5 else 7
    views = {}

    def view(i):
        if i not in views:
            if len(views) > 48:
                views.pop(next(iter(views)))
            views[i] = View(chunk, i, scale)
        return views[i]

    if method == "patchmatch":
        log("Build Depth Maps: %d cameras, quality=%s (scale %.3g), PatchMatch with normals" % (len(aligned), quality, scale))
    else:
        log("Build Depth Maps: %d cameras, quality=%s (scale %.3g), %d planes" % (len(aligned), quality, scale, n_planes))
    jobs = []
    for i in aligned:
        p = chunk.path("depth", "%05d.npz" % i)
        if (os.path.exists(p) and not force) or not nbmap.get(i):
            continue
        jobs.append((chunk.project.dir, chunk.project.chunks.index(chunk), i, nbmap[i], scale, n_planes, win, method))
    from .learned.mvs_gpu import gpu_available
    use_gpu = gpu_available() and gpu is not False
    workers = workers or max(1, min((os.cpu_count() or 2) - 1, 8))
    if use_gpu:
        import torch
        from .util import fast_mode
        log("  using GPU: %s" % torch.cuda.get_device_name(0))
        from .learned.mvs_gpu import compute_depth_gpu
        if fast_mode():
            # CPU threads decode + undistort upcoming views and write results while the GPU sweeps planes
            from concurrent.futures import ThreadPoolExecutor
            loader = ThreadPoolExecutor(max(2, min(6, (os.cpu_count() or 4) // 2)))
            writer = ThreadPoolExecutor(2)
            futs, writes = {}, []

            def vfut(i):
                if i not in futs:
                    futs[i] = loader.submit(View, chunk, i, scale)
                return futs[i]
            lookahead = 3
            for k, (_, _, i, nb, sc, npl, wn, meth) in enumerate(jobs):
                for (_, _, i2, nb2, _, _, _, _) in jobs[k:k + lookahead + 1]:
                    vfut(i2); [vfut(j) for j in nb2]
                ref = vfut(i).result()
                rng = depth_range(chunk, i, ref)
                if rng is not None:
                    d, conf = _compute(meth, ref, [vfut(j).result() for j in nb], rng[0], rng[1], npl, wn, gpu=True)
                    writes.append(writer.submit(save_depth, chunk.path("depth", "%05d.npz" % i), d, conf, ref.K, sc))
                    log("  depth map %d/%d (%s): %.0f%% valid" % (k + 1, len(jobs), chunk.cameras[i].label,
                                                                 100 * (d > 0).mean()))
                # release views no longer needed by upcoming jobs
                keep = set()
                for (_, _, i2, nb2, _, _, _, _) in jobs[k + 1:k + lookahead + 2]:
                    keep.add(i2); keep.update(nb2)
                for j in [j for j in futs if j not in keep]:
                    del futs[j]
            for w in writes:
                w.result()
            loader.shutdown(); writer.shutdown()
        else:
            for k, (_, _, i, nb, sc, npl, wn, meth) in enumerate(jobs):
                ref = view(i)
                rng = depth_range(chunk, i, ref)
                if rng is None:
                    continue
                d, conf = _compute(meth, ref, [view(j) for j in nb], rng[0], rng[1], npl, wn, gpu=True)
                save_depth(chunk.path("depth", "%05d.npz" % i), d, conf, ref.K, sc)
                log("  depth map %d/%d (%s): %.0f%% valid" % (k + 1, len(jobs), chunk.cameras[i].label, 100 * (d > 0).mean()))
    elif workers > 1 and len(jobs) > 2 and method != "patchmatch":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            for k, msg in enumerate(ex.map(_depth_job, jobs)):
                log("  depth map %d/%d %s" % (k + 1, len(jobs), msg))
    else:
        for k, jb in enumerate(jobs):
            log("  depth map %d/%d %s" % (k + 1, len(jobs), _depth_job(jb, view)))
    chunk.stats["depth_maps"] = dict(quality=quality, filter=filter_mode, cameras=len(aligned), method=method)
    from .util import fast_mode
    have_f = all(os.path.exists(chunk.path("depth", "%05d_f.npz" % i)) for i in aligned
                 if os.path.exists(chunk.path("depth", "%05d.npz" % i)))
    if not jobs and not force and have_f and prev_filter == filter_mode and fast_mode():
        log("  all depth maps and their %s filtering already exist - reused (tick 'Recompute existing' to redo)"
            % filter_mode)
    else:
        log("  depth maps done; filtering (%s): keeping only depths confirmed by %d+ neighbouring photos..." %
            (filter_mode, FILTER[filter_mode]))
        filter_depth_maps(chunk, filter_mode, nbmap, views=view)
    chunk.add_log("Build Depth Maps", time.time() - t0, quality=quality, filter=filter_mode, method=method)


def _depth_job(args, view_fn=None):
    """Compute + save one depth map. Runs in a worker process (view_fn=None) or in-process with a view cache."""
    from .project import Project
    proj_dir, chunk_idx, i, nb, scale, n_planes, win, meth = args
    ch = Project.open(proj_dir).chunk(chunk_idx)
    if view_fn is None:
        view_fn = lambda j: View(ch, j, scale)
    ref = view_fn(i)
    rng = depth_range(ch, i, ref)
    if rng is None:
        return "%d: no depth range" % i
    nbrs = [view_fn(j) for j in nb]
    d, conf = _compute(meth, ref, nbrs, rng[0], rng[1], n_planes, win, gpu=False)
    save_depth(ch.path("depth", "%05d.npz" % i), d, conf, ref.K, scale)
    return "(%s): %.0f%% valid" % (ch.cameras[i].label, 100 * (d > 0).mean())


def _compute(method, ref, nbrs, zmin, zmax, n_planes, win, gpu):
    if method == "patchmatch":
        from .learned.patchmatch import patchmatch_depth
        return patchmatch_depth(ref, nbrs, zmin, zmax, prefer="auto" if gpu else "torch")
    if gpu:
        from .learned.mvs_gpu import compute_depth_gpu
        return compute_depth_gpu(ref, nbrs, zmin, zmax, n_planes, win)
    return compute_depth(ref, nbrs, zmin, zmax, n_planes, win)


def save_depth(path, d, conf, K, scale):
    from .util import fast_mode
    arrs = dict(depth=d.astype(np.float16), conf=(conf * 255).clip(0, 255).astype(np.uint8), K=K, scale=scale)
    # uncompressed in fast mode: zlib on every map cost more CPU than the GPU sweep itself
    (np.savez if fast_mode() else np.savez_compressed)(path, **arrs)
    fp = path[:-4] + "_f.npz"
    if os.path.exists(fp):
        os.remove(fp)                                   # stale filter result


def load_depth(chunk, i, filtered=True):
    p = os.path.join(chunk.dir, "depth", "%05d.npz" % i)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    out = dict(depth=d["depth"].astype(np.float32), conf=d["conf"], K=d["K"])
    if filtered:
        fp = p[:-4] + "_f.npz"
        if os.path.exists(fp):
            f = np.load(fp)
            out["depth"] = np.where(f["fmask"], out["depth"], 0)
            out["nconsist"] = f["nconsist"]
        elif "fmask" in d:
            out["depth"] = np.where(d["fmask"], out["depth"], 0)
            out["nconsist"] = d["nconsist"]
    return out


def _project_into(Xw, R, t, K):
    Xc = Xw @ R.T + t
    z = Xc[..., 2]
    uv = Xc @ K.T
    return uv[..., 0] / np.maximum(z, 1e-9), uv[..., 1] / np.maximum(z, 1e-9), z


def _consistency(Di, Ki, Ri, ti, nbrs, need, rel_tol, px_tol):
    """nbrs: list of (Dj, Kj, Rj, tj). Returns (fmask, count) exactly as the original loop."""
    h, w = Di.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rays = np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(Ki).T
    Xw = (rays * Di[..., None] - ti) @ Ri
    cnt = np.zeros((h, w), np.uint8)
    for Dj, Kj, Rj, tj in nbrs:
        u, v, z = _project_into(Xw, Rj, tj, Kj)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inb = (ui >= 0) & (ui < Dj.shape[1]) & (vi >= 0) & (vi < Dj.shape[0]) & (z > 0)
        dj = np.zeros_like(Di)
        dj[inb] = Dj[vi[inb], ui[inb]]
        ok = inb & (dj > 0) & (Di > 0)
        rays_j = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(Kj).T
        Xj = (rays_j * dj[..., None] - tj) @ Rj
        u2, v2, z2 = _project_into(Xj, Ri, ti, Ki)
        reproj = np.hypot(u2 - xs, v2 - ys)
        rel = np.abs(z2 - Di) / np.maximum(Di, 1e-9)
        cnt += (ok & (reproj < px_tol * 2) & (rel < rel_tol)).astype(np.uint8)
    return (cnt >= need) & (Di > 0), cnt


def _raw_depth(path):
    d = np.load(path)
    return d["depth"].astype(np.float32), d["K"]


def _filter_job(args):
    ddir, i, nb, poses, need, rel_tol, px_tol = args
    Di, Ki = _raw_depth(os.path.join(ddir, "%05d.npz" % i))
    Ri, ti = poses[i]
    nbrs = []
    for j in nb:
        Dj, Kj = _raw_depth(os.path.join(ddir, "%05d.npz" % j))
        nbrs.append((Dj, Kj, poses[j][0], poses[j][1]))
    fmask, cnt = _consistency(Di, Ki, Ri, ti, nbrs, need, rel_tol, px_tol)
    np.savez(os.path.join(ddir, "%05d_f.npz" % i), fmask=fmask, nconsist=cnt)
    return i


def filter_depth_maps(chunk, mode="moderate", nbmap=None, views=None, rel_tol=0.01, px_tol=1.0):
    from .util import fast_mode, n_workers
    need = FILTER[mode]
    nbmap = nbmap or select_neighbors(chunk)
    aligned = [i for i in chunk.aligned_cameras() if os.path.exists(chunk.path("depth", "%05d.npz" % i))]
    poses = {i: (cm.rodrigues_to_R(chunk.cameras[i].rvec)[0], np.asarray(chunk.cameras[i].tvec)) for i in aligned}
    ddir = os.path.join(chunk.dir, "depth")
    if fast_mode():
        jobs = [(ddir, i, [j for j in nbmap.get(i, []) if j in poses],
                 {k: poses[k] for k in [i] + [j for j in nbmap.get(i, []) if j in poses]}, need, rel_tol, px_tol)
                for i in aligned]
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_workers()) as ex:
            for k, _ in enumerate(ex.map(_filter_job, jobs, chunksize=2)):
                if (k + 1) % max(1, len(jobs) // 10) == 0:
                    log("  filtered %d/%d depth maps" % (k + 1, len(jobs)))
        log("  depth filtering (%s) done" % mode)
        return
    raw = {i: load_depth(chunk, i, filtered=False) for i in aligned}
    for i in aligned:
        Ri, ti = poses[i]
        nbrs = [(raw[j]["depth"], raw[j]["K"], poses[j][0], poses[j][1]) for j in nbmap.get(i, []) if j in raw]
        fmask, cnt = _consistency(raw[i]["depth"], raw[i]["K"], Ri, ti, nbrs, need, rel_tol, px_tol)
        fp = chunk.path("depth", "%05d_f.npz" % i)
        if os.path.exists(fp):
            os.remove(fp)
        p = chunk.path("depth", "%05d.npz" % i)
        d = dict(np.load(p))
        d["fmask"], d["nconsist"] = fmask, cnt
        np.savez_compressed(p, **d)
    log("  depth filtering (%s) done" % mode)


def _cloud_view(chunk, i, stride):
    d = load_depth(chunk, i)
    if d is None:
        return None
    cam = chunk.cameras[i]
    sen = chunk.sensors[cam.sensor]
    D, K = d["depth"], d["K"]
    h, w = D.shape
    scale = w / sen.width
    img = read_image(chunk.abspath(cam))
    m1, m2, _ = cm.undistort_maps(sen.calib(), sen.width, sen.height, scale)
    col = cv2.remap(img, m1, m2, cv2.INTER_AREA)
    R, t = cm.rodrigues_to_R(cam.rvec)[0], np.asarray(cam.tvec)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    Xc = (np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(K).T) * D[..., None]
    dx = np.zeros_like(Xc); dy = np.zeros_like(Xc)
    dx[:, 1:-1] = Xc[:, 2:] - Xc[:, :-2]
    dy[1:-1] = Xc[2:] - Xc[:-2]
    nrm = np.cross(dx, dy)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
    flip = (nrm * Xc).sum(-1) > 0
    nrm[flip] *= -1
    valid = D > 0
    valid[:, [0, -1]] = False; valid[[0, -1], :] = False
    vd = valid.copy()
    vd[1:-1, 1:-1] &= valid[2:, 1:-1] & valid[:-2, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
    valid = vd & (np.abs((nrm * Xc).sum(-1)) / (np.linalg.norm(Xc, axis=-1) + 1e-9) > 0.15)
    if stride > 1:
        sm = np.zeros_like(valid); sm[::stride, ::stride] = True; valid &= sm
    Xw = (Xc[valid] - t) @ R
    return (Xw.astype(np.float32), col[valid][:, ::-1], (nrm[valid] @ R).astype(np.float32),
            d.get("nconsist", np.ones_like(D, np.uint8))[valid], scale)


def build_point_cloud(chunk, point_spacing=None, stride=1, min_conf=0):
    """Fuse filtered depth maps into a coloured dense cloud with normals and confidence."""
    t0 = time.time()
    scale = None
    P, Cc, N, Q = [], [], [], []
    from .util import fast_mode, n_workers
    cams = chunk.aligned_cameras()
    log("Build Point Cloud: back-projecting %d filtered depth maps into 3D (with colours and normals)..." % len(cams))
    est = _estimate_raw_points(chunk, cams)
    if fast_mode() and est > STREAM_POINTS:
        return _build_point_cloud_streaming(chunk, cams, stride, point_spacing, min_conf, est, t0)
    if fast_mode():
        from concurrent.futures import ThreadPoolExecutor
        outs = []
        with ThreadPoolExecutor(n_workers(8)) as ex:          # numpy / OpenCV release the GIL
            for k, o in enumerate(ex.map(lambda i: _cloud_view(chunk, i, stride), cams)):
                outs.append(o)
                if (k + 1) % max(1, len(cams) // 10) == 0:
                    log("  %d/%d depth maps back-projected" % (k + 1, len(cams)))
    else:
        outs = [_cloud_view(chunk, i, stride) for i in cams]
    for o in outs:
        if o is None:
            continue
        P.append(o[0]); Cc.append(o[1]); N.append(o[2]); Q.append(o[3]); scale = o[4]
    P, Cc, N, Q = np.concatenate(P), np.concatenate(Cc), np.concatenate(N), np.concatenate(Q)
    P, Cc, N, Q = crop_to_region(chunk, P, Cc, N, Q)
    if point_spacing is None:
        point_spacing = estimate_gsd(chunk, scale) * 1.0
    log("  merging %s raw points on a %.4g grid (duplicates from overlapping photos are averaged)..." %
        ("{:,}".format(len(P)), point_spacing))
    P, Cc, N, Q = voxel_merge(P, Cc, N, Q, point_spacing)
    if min_conf:
        m = Q >= min_conf
        P, Cc, N, Q = P[m], Cc[m], N[m], Q[m]
    from .io import write_ply
    log("  writing point_cloud.ply (%s points)..." % "{:,}".format(len(P)))
    write_ply(chunk.path("point_cloud.ply"), P, colors=Cc, normals=N, extra={"confidence": Q.astype(np.uint8)})
    chunk.stats["point_cloud"] = dict(points=int(len(P)), spacing=float(point_spacing))
    chunk.add_log("Build Point Cloud", time.time() - t0, points=int(len(P)))
    log("Point cloud: %d points (spacing %.4g)" % (len(P), point_spacing))
    return P, Cc, N, Q


STREAM_POINTS = 40_000_000      # above this many raw points the cloud is fused out-of-core


def _estimate_raw_points(chunk, cams):
    for i in cams[:: max(1, len(cams) // 10)]:
        d = load_depth(chunk, i)
        if d is not None:
            return int(len(cams) * d["depth"].size * 0.7)
    return 0


def _build_point_cloud_streaming(chunk, cams, stride, point_spacing, min_conf, est, t0):
    """Out-of-core fusion for big projects (constant memory, identical result to the in-memory path).
    Pass 1 back-projects views in order and appends the points to slab files on disk (slabs = ranges of the
    first voxel index); pass 2 voxel-merges each slab with the global voxel origin, so voxels never straddle
    slabs and the concatenated slabs are in the same order as one global merge; the PLY is written streaming."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    from .util import n_workers
    if point_spacing is None:
        point_spacing = estimate_gsd(chunk, None) * 1.0
    size = point_spacing
    tmp = chunk.path("_cloud_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    # slab ranges over the first voxel index, from the region (or tie points)
    if chunk.region:
        c, s, R = region_box(chunk)
        corners = c + (np.array([[i, j, k] for i in (-.5, .5) for j in (-.5, .5) for k in (-.5, .5)]) * s) @ R
    else:
        corners = chunk.load_sparse()["points"]
    k0lo = int(np.floor(corners[:, 0].min() / size)) - 1
    k0hi = int(np.floor(corners[:, 0].max() / size)) + 1
    nslab = int(np.clip(np.ceil(est / 12e6), 2, 256))
    log("  large cloud (~%s raw points): fusing out-of-core in %d slabs (temporary files in %s)" %
        ("{:,}".format(est), nslab, tmp))
    files = [[open(os.path.join(tmp, "%03d_%s.bin" % (k, f)), "wb") for f in "PCNQ"] for k in range(nslab)]
    counts = np.zeros(nslab, np.int64)
    kmin = np.array([np.iinfo(np.int64).max] * 3); kmax = np.array([np.iinfo(np.int64).min] * 3)
    total = 0
    batch = max(4, n_workers(8) * 2)
    with ThreadPoolExecutor(n_workers(8)) as ex:
        done = 0
        for b0 in range(0, len(cams), batch):                 # bounded batches keep memory constant
            for o in ex.map(lambda i: _cloud_view(chunk, i, stride), cams[b0:b0 + batch]):
                done += 1
                if o is None:
                    continue
                P, Cc, N, Q = crop_to_region(chunk, o[0], o[1], o[2], o[3])
                if len(P) == 0:
                    continue
                key = np.floor(P / size).astype(np.int64)
                kmin = np.minimum(kmin, key.min(0)); kmax = np.maximum(kmax, key.max(0))
                sl = np.clip((key[:, 0] - k0lo) * nslab // max(1, k0hi - k0lo + 1), 0, nslab - 1)
                for k in np.unique(sl):
                    m = sl == k
                    fs = files[k]
                    fs[0].write(np.ascontiguousarray(P[m], np.float32).tobytes())
                    fs[1].write(np.ascontiguousarray(Cc[m], np.uint8).tobytes())
                    fs[2].write(np.ascontiguousarray(N[m], np.float32).tobytes())
                    fs[3].write(np.ascontiguousarray(Q[m], np.uint8).tobytes())
                    counts[k] += m.sum()
                total += len(P)
            log("  %d/%d depth maps back-projected (%s points so far)" % (done, len(cams), "{:,}".format(total)))
    for fs in files:
        for f in fs:
            f.close()
    log("  merging %s raw points on a %.4g grid, slab by slab (duplicates from overlapping photos are averaged)..."
        % ("{:,}".format(total), size))
    dims = kmax - kmin + 1
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
              ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("confidence", "u1")]
    merged = []
    n_out = 0
    for k in range(nslab):
        if counts[k] == 0:
            continue
        base = os.path.join(tmp, "%03d_" % k)
        P = np.fromfile(base + "P.bin", np.float32).reshape(-1, 3)
        Cc = np.fromfile(base + "C.bin", np.uint8).reshape(-1, 3)
        N = np.fromfile(base + "N.bin", np.float32).reshape(-1, 3)
        Q = np.fromfile(base + "Q.bin", np.uint8)
        Pm, Cm, Nm, Qm = voxel_merge(P, Cc, N, Q, size, kmin, dims)
        del P, Cc, N, Q
        if min_conf:
            m = Qm >= min_conf
            Pm, Cm, Nm, Qm = Pm[m], Cm[m], Nm[m], Qm[m]
        arr = np.zeros(len(Pm), dtype=fields)
        arr["x"], arr["y"], arr["z"] = Pm.T
        arr["nx"], arr["ny"], arr["nz"] = Nm.T
        arr["red"], arr["green"], arr["blue"] = Cm[:, 0], Cm[:, 1], Cm[:, 2]
        arr["confidence"] = Qm.astype(np.uint8)
        mp = base + "merged.bin"
        arr.tofile(mp)
        merged.append(mp); n_out += len(arr)
        for f in "PCNQ":
            os.remove(base + f + ".bin")
        log("  slab %d/%d merged: %s points" % (k + 1, nslab, "{:,}".format(len(arr))))
    tmap = {"<f4": "float", "u1": "uchar"}
    hdr = ["ply", "format binary_little_endian 1.0", "element vertex %d" % n_out]
    hdr += ["property %s %s" % (tmap[t], name) for name, t in fields]
    hdr.append("end_header")
    log("  writing point_cloud.ply (%s points)..." % "{:,}".format(n_out))
    with open(chunk.path("point_cloud.ply"), "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode())
        for mp in merged:
            with open(mp, "rb") as g:
                shutil.copyfileobj(g, f, 64 << 20)
    shutil.rmtree(tmp, ignore_errors=True)
    chunk.stats["point_cloud"] = dict(points=int(n_out), spacing=float(point_spacing))
    chunk.add_log("Build Point Cloud", time.time() - t0, points=int(n_out))
    log("Point cloud: %d points (spacing %.4g)" % (n_out, point_spacing))
    return None, None, None, None


def estimate_gsd(chunk, scale):
    """Median ground sampling distance (world units per depth-map pixel)."""
    vals = []
    for i in chunk.aligned_cameras()[::max(1, len(chunk.aligned_cameras()) // 20)]:
        d = load_depth(chunk, i)
        if d is None:
            continue
        z = d["depth"][d["depth"] > 0]
        if len(z):
            vals.append(np.median(z) / d["K"][0, 0])
    return float(np.median(vals)) if vals else 0.01


def region_box(chunk):
    r = chunk.region
    return np.array(r["center"]), np.array(r["size"]), np.array(r["R"])


def crop_to_region(chunk, P, *others):
    if not chunk.region:
        return (P,) + others
    c, s, R = region_box(chunk)
    L = (P - c) @ R.T
    m = (np.abs(L) <= s / 2).all(1)
    return (P[m],) + tuple(o[m] for o in others)


def voxel_merge(P, C, N, Q, size, kmin=None, dims=None):
    """Average all points falling in the same voxel. kmin / dims: global voxel-index origin and extent when
    merging one slab of a larger cloud (gives exactly the global result, in the global order)."""
    key = np.floor(P / size).astype(np.int64)
    key -= key.min(0) if kmin is None else kmin
    dims = key.max(0) + 1 if dims is None else dims
    lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
    uniq, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
    def avg(a):
        out = np.zeros((len(uniq), a.shape[1]))
        for k in range(a.shape[1]):
            out[:, k] = np.bincount(inv, a[:, k].astype(np.float64), minlength=len(uniq))
        return out / cnt[:, None]
    Pm = avg(P).astype(np.float32)
    Cm = avg(C).round().clip(0, 255).astype(np.uint8)
    Nm = avg(N); Nm /= np.linalg.norm(Nm, axis=1, keepdims=True) + 1e-12
    Qm = np.minimum(cnt + np.bincount(inv, Q.astype(float), minlength=len(uniq)) / cnt, 255)
    return Pm, Cm, Nm.astype(np.float32), Qm
