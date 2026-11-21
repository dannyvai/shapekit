"""Build Texture (generic mapping, mosaic blending).

1. visibility per camera via z-buffer
2. per-face best view (projected area x viewing angle x image quality), label smoothing
3. charts = connected faces sharing a source view; UVs = projection into that view
4. charts packed into an atlas, pixels copied from undistorted source frames
5. per-camera colour gain correction (like Calibrate Colors) to hide exposure seams
"""
import time
import numpy as np
import cv2
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import camera as cm
from . import raster
from .mesh import load_mesh, face_normals
from .io import write_obj, write_glb, write_ply
from .util import log, read_image


def face_adjacency(F):
    nf = len(F)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(1)
    fid = np.tile(np.arange(nf), 3)
    key = e[:, 0].astype(np.int64) * (F.max() + 1) + e[:, 1]
    o = np.argsort(key)
    key, fid = key[o], fid[o]
    same = key[1:] == key[:-1]
    a, b = fid[:-1][same], fid[1:][same]
    return a, b


def build_texture(chunk, size=8192, vis_scale=0.5, color_correction=True, smooth_iters=6, jpeg_quality=92,
                  blending="mosaic"):
    """blending: 'mosaic' (each face from its single best photo, per-photo gain) or 'multiband' (mosaic for
    detail + low frequencies blended across all photos that see the surface: seams and exposure steps
    between charts are levelled out)."""
    t0 = time.time()
    V, F, VC = load_mesh(chunk)
    nf = len(F)
    fn, farea = face_normals(V, F)
    cen = V[F].mean(1)
    cams = chunk.aligned_cameras()
    rows, cols, vals = [], [], []
    vis_rgb = {}
    try:
        from .learned.raster_gpu import available as _gpu
        dev = "GPU" if _gpu() else "CPU"
    except Exception:
        dev = "CPU"
    from .util import fast_mode, n_workers
    vgpu = None
    if blending == "multiband":
        color_correction = True                 # needs the per-view visibility lists kept for colour sampling
    if fast_mode() and dev == "GPU":
        from .learned.texture_gpu import VisibilityGPU
        vgpu = VisibilityGPU(V, F, cen, fn, farea)
        dev = "GPU (resident mesh)"
    log("Build Texture: %d faces, %d cameras (visibility on %s)" % (nf, len(cams), dev))
    from . import masks as MS
    mcams = set(MS.masked_cameras(chunk))

    def apply_mask(i, vis, score, pu_v, pv_v):
        """Drop faces whose centre falls inside the photo's mask (undistorted to the visibility image)."""
        if i not in mcams:
            return vis, score, pu_v, pv_v
        sen = chunk.sensors[chunk.cameras[i].sensor]
        mk = MS.load_mask(chunk, i)
        if mk is None:
            return vis, score, pu_v, pv_v
        m1, m2, _ = cm.undistort_maps(sen.calib(), sen.width, sen.height, vis_scale)
        mu = cv2.remap(mk.astype(np.uint8), m1, m2, cv2.INTER_NEAREST) > 0
        h_, w_ = mu.shape
        keep = mu[np.clip(pv_v.astype(int), 0, h_ - 1), np.clip(pu_v.astype(int), 0, w_ - 1)]
        return vis[keep], score[keep], pu_v[keep], pv_v[keep]
    for k, i in enumerate(cams):
        if vgpu is not None:
            cam = chunk.cameras[i]
            vis, score, pu_v, pv_v = vgpu.camera(cam, chunk.sensors[cam.sensor], vis_scale)
            vis, score, pu_v, pv_v = apply_mask(i, vis, score, pu_v, pv_v)
            rows.append(vis); cols.append(np.full(len(vis), k)); vals.append(score)
            if color_correction:
                vis_rgb[k] = (vis, pu_v, pv_v)
            if k % 10 == 0:
                log("  visibility %d/%d" % (k + 1, len(cams)))
            continue
        cam = chunk.cameras[i]
        sen = chunk.sensors[cam.sensor]
        Kf = cm.K_matrix(sen.intr)
        Ks = cm.K_matrix(sen.intr, vis_scale)
        w, h = int(round(sen.width * vis_scale)), int(round(sen.height * vis_scale))
        uv, z = raster.project_vertices(V, cam.rvec, cam.tvec, Ks)
        ids, zb = raster.rasterize(uv, z, F, w, h)
        R = cm.rodrigues_to_R(cam.rvec)[0]
        Xc = cen @ R.T + cam.tvec
        zc = Xc[:, 2]
        pu = Xc[:, 0] / np.maximum(zc, 1e-9) * Ks[0, 0] + Ks[0, 2]
        pv = Xc[:, 1] / np.maximum(zc, 1e-9) * Ks[1, 1] + Ks[1, 2]
        m = 0.01 * w
        inb = (zc > 0) & (pu >= m) & (pu < w - m) & (pv >= m) & (pv < h - m)
        f_in = np.where(inb)[0]
        zz = zb[pv[f_in].astype(int), pu[f_in].astype(int)]
        vis = f_in[zc[f_in] <= zz * 1.01 + 1e-6]
        hit = np.unique(ids[ids >= 0])
        vis = np.union1d(vis, hit)
        # viewing angle
        dirv = (cm.camera_center(cam.rvec, cam.tvec) - cen[vis])
        dist = np.linalg.norm(dirv, axis=1)
        cosang = (fn[vis] * dirv).sum(1) / np.maximum(dist, 1e-12)
        good = cosang > 0.1
        vis, cosang, dist = vis[good], cosang[good], dist[good]
        # projected area at full resolution ~ area * cos * f^2 / d^2
        area_px = farea[vis] * cosang * (Kf[0, 0] / dist) ** 2
        # penalise distance from image centre slightly (lens falloff / distortion)
        rc = np.hypot((pu[vis] - w / 2) / (w / 2), (pv[vis] - h / 2) / (h / 2))
        q = cam.quality if cam.quality else 1.0
        score = area_px * cosang * (1 - 0.25 * np.clip(rc, 0, 1.4)) * (0.3 + 0.7 * q)
        vis, score, pu_k, pv_k = apply_mask(i, vis, score, pu[vis], pv[vis])
        rows.append(vis); cols.append(np.full(len(vis), k)); vals.append(score)
        if color_correction:
            vis_rgb[k] = (vis, pu_k, pv_k)
        if k % 10 == 0:
            log("  visibility %d/%d" % (k + 1, len(cams)))
    S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nf, len(cams)))
    best = np.asarray(S.argmax(axis=1)).ravel()
    bestv = np.asarray(S.max(axis=1).todense()).ravel()
    label = np.where(bestv > 0, best, -1)

    # ---- label smoothing (majority of neighbours if their view is nearly as good)
    a, b = face_adjacency(F)
    for it in range(smooth_iters):
        votes = {}
        nb_lab = np.concatenate([label[b], label[a]])
        tgt = np.concatenate([a, b])
        ok = nb_lab >= 0
        key = tgt[ok].astype(np.int64) * (len(cams) + 1) + nb_lab[ok]
        uk, cnt = np.unique(key, return_counts=True)
        f_of, l_of = uk // (len(cams) + 1), uk % (len(cams) + 1)
        o = np.lexsort((-cnt, f_of))
        f_of, l_of, cnt = f_of[o], l_of[o], cnt[o]
        first = np.concatenate([[True], f_of[1:] != f_of[:-1]])
        f_of, l_of, cnt = f_of[first], l_of[first], cnt[first]
        cand = (cnt >= 2) & (l_of != label[f_of])
        f_c, l_c = f_of[cand], l_of[cand]
        sc = np.asarray(S[f_c, l_c]).ravel()
        cur = np.where(label[f_c] >= 0, bestv[f_c], 0)
        switch = (sc >= 0.35 * cur) | (label[f_c] < 0)
        # unlabeled faces simply adopt the neighbour label
        label[f_c[switch]] = l_c[switch]
    log("  labelled %.1f%% of faces" % (100 * (label >= 0).mean()))

    # ---- colour correction: per camera gain vs mesh vertex colours (median ratio)
    gains = np.ones((len(cams), 3))
    if color_correction and VC is not None:
        ref = VC[F].mean(1).astype(np.float64) + 1

        def gain(k):
            i = cams[k]
            vis, pu, pv = vis_rgb[k]
            sel = np.where(label[vis] == k)[0]
            if len(sel) < 200:
                sel = np.arange(len(vis))
            if len(sel) < 50:
                return None
            sel = sel[:: max(1, len(sel) // 5000)]
            img = read_image(chunk.abspath(chunk.cameras[i]), scale=vis_scale)
            c = img[pv[sel].astype(int).clip(0, img.shape[0] - 1), pu[sel].astype(int).clip(0, img.shape[1] - 1), ::-1]
            return np.clip(np.median(ref[vis[sel]] / (c + 1.0), axis=0), 0.6, 1.6)
        if fast_mode():
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(n_workers(8)) as ex:
                res = list(ex.map(gain, range(len(cams))))
        else:
            res = [gain(k) for k in range(len(cams))]
        for k, g in enumerate(res):
            if g is not None:
                gains[k] = g
        gains /= np.median(gains, axis=0)

    # ---- charts
    same = label[a] == label[b]
    A = sp.coo_matrix((np.ones(same.sum()), (a[same], b[same])), shape=(nf, nf))
    nchart, chart = connected_components(A, directed=False)
    log("  %d charts" % nchart)
    corner_uv_px = np.zeros((nf, 3, 2))
    boxes = np.zeros((nchart, 4))
    chart_cam = np.full(nchart, -1)
    chart_cam[chart] = label
    for k, i in enumerate(cams):
        fsel = np.where(label == k)[0]
        if len(fsel) == 0:
            continue
        cam = chunk.cameras[i]
        sen = chunk.sensors[cam.sensor]
        uv, z = raster.project_vertices(V, cam.rvec, cam.tvec, cm.K_matrix(sen.intr))
        corner_uv_px[fsel] = uv[F[fsel]]
    pmin = corner_uv_px.min(1); pmax = corner_uv_px.max(1)
    bx0 = np.full(nchart, np.inf); by0 = np.full(nchart, np.inf); bx1 = np.full(nchart, -np.inf); by1 = np.full(nchart, -np.inf)
    np.minimum.at(bx0, chart, pmin[:, 0]); np.minimum.at(by0, chart, pmin[:, 1])
    np.maximum.at(bx1, chart, pmax[:, 0]); np.maximum.at(by1, chart, pmax[:, 1])
    pad = 3
    boxes = np.stack([np.floor(bx0) - pad, np.floor(by0) - pad, np.ceil(bx1) + pad, np.ceil(by1) + pad], 1)
    labeled = chart_cam >= 0
    bw = boxes[:, 2] - boxes[:, 0]; bh = boxes[:, 3] - boxes[:, 1]
    total = (bw * bh)[labeled].sum()
    s = min(1.0, np.sqrt(0.70 * size * size / max(total, 1)))
    for attempt in range(20):
        pos = _shelf_pack(np.ceil(bw * s) + 2, np.ceil(bh * s) + 2, labeled, size)
        if pos is not None:
            break
        s *= 0.93
    log("  atlas %dx%d, chart scale %.3f" % (size, size, s))
    atlas = np.full((size, size, 3), 127, np.uint8)
    # gray patch for unlabelled faces

    def fill(k):
        i = cams[k]
        cs = np.where(labeled & (chart_cam == k))[0]
        if len(cs) == 0:
            return
        cam = chunk.cameras[i]
        sen = chunk.sensors[cam.sensor]
        img = read_image(chunk.abspath(cam))
        m1, m2, _ = cm.undistort_maps(sen.calib(), sen.width, sen.height, 1.0)
        und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if color_correction:
            und = np.clip(und.astype(np.float32) * gains[k][::-1][None, None], 0, 255).astype(np.uint8)
        for c in cs:
            x0, y0, x1, y1 = boxes[c].astype(int)
            ax, ay = pos[c]
            tw, th = int(np.ceil(bw[c] * s)), int(np.ceil(bh[c] * s))
            if tw <= 0 or th <= 0:
                continue
            # crop with border handling
            M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
            patch = cv2.warpAffine(und, M, (tw + 2, th + 2), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            atlas[ay:ay + th + 2, ax:ax + tw + 2] = patch[:min(th + 2, size - ay), :min(tw + 2, size - ax)]
    if fast_mode():
        # charts own disjoint atlas rectangles, so cameras can be filled concurrently
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(n_workers(8)) as ex:
            list(ex.map(fill, range(len(cams))))
    else:
        for k in range(len(cams)):
            fill(k)
    # UVs
    uv_px = np.zeros((nf, 3, 2))
    fl = label >= 0
    c_of = chart[fl]
    uv_px[fl] = (corner_uv_px[fl] - boxes[c_of, None, :2]) * s + np.array(pos)[c_of][:, None, :]
    if blending == "multiband":
        _seam_leveling(chunk, V, F, cams, label, S, vis_rgb, gains, vis_scale, atlas, uv_px, fl)
    uv_px[~fl] = [2, 2]
    atlas[0:5, 0:5] = 127
    UV = uv_px.reshape(-1, 2) / size
    UV[:, 1] = 1 - UV[:, 1]
    face_uv = np.arange(nf * 3).reshape(nf, 3)
    out = chunk.path("model", "model.obj")
    tex = chunk.path("model", "model_texture.jpg")
    cv2.imwrite(tex, atlas[:, :, ::-1] if False else atlas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    write_obj(out, V, F, UV, face_uv, texture_file=tex)
    np.savez_compressed(chunk.path("model", "uv.npz"), uv=UV.astype(np.float32), face_uv=face_uv)
    chunk.stats["texture"] = dict(size=size, charts=int(nchart), scale=float(s), blending=blending,
                                  textured_faces_pct=float(100 * fl.mean()))
    chunk.add_log("Build Texture", time.time() - t0, size=size, charts=int(nchart))
    log("Texture written: %s" % out)
    return out


def _shelf_pack(w, h, use, size):
    order = np.argsort(-h)
    pos = np.zeros((len(w), 2), int)
    x = y = shelf_h = 0
    y = 6  # keep corner for the gray patch
    x = 6
    for c in order:
        if not use[c]:
            continue
        cw, ch = int(w[c]), int(h[c])
        if cw > size or ch > size:
            return None
        if x + cw > size:
            y += shelf_h; x = 0; shelf_h = 0
        if y + ch > size:
            return None
        pos[c] = (x, y)
        x += cw
        shelf_h = max(shelf_h, ch)
    return pos


def _bilinear(img, x, y):
    """img (h,w,c) float32 sampled at float pixel coords (in range) -> (n,c)."""
    h, w = img.shape[:2]
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 2); y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 2)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) +
            (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def _seam_leveling(chunk, V, F, cams, label, S, vis_rgb, gains, vis_scale, atlas, uv_px, fl, sigma=2.0):
    """Two-band blending in the atlas. Low band: every vertex gets the score-weighted mean of its (low-pass,
    gain-corrected) colour over all photos that see it; each face corner's own-photo colour is pulled to that
    mean by a smooth correction field interpolated across the face and added to the mosaic (high band kept
    from the single best photo, so detail stays sharp)."""
    from concurrent.futures import ThreadPoolExecutor
    from .util import n_workers
    nv, nf = len(V), len(F)
    bsum = np.zeros((nv, 3)); wsum = np.zeros(nv)
    own = np.full((nf, 3, 3), np.nan)
    Fi = np.asarray(F, np.int64)

    def view(k):
        i = cams[k]
        cam = chunk.cameras[i]; sen = chunk.sensors[cam.sensor]
        vis = vis_rgb[k][0]
        if len(vis) == 0:
            return k, None
        img = read_image(chunk.abspath(cam))
        m1, m2, Ks = cm.undistort_maps(sen.calib(), sen.width, sen.height, vis_scale)
        und = cv2.remap(img, m1, m2, cv2.INTER_AREA)
        und = cv2.GaussianBlur(und.astype(np.float32), (0, 0), sigma)
        fv = Fi[vis]
        vids = np.unique(fv)
        uv, z = raster.project_vertices(V[vids], cam.rvec, cam.tvec, Ks)
        h, w = und.shape[:2]
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] <= h - 1)
        col = np.full((len(vids), 3), np.nan)
        if ok.any():
            col[ok] = _bilinear(und, uv[ok, 0], uv[ok, 1])[:, ::-1] * gains[k][None]
        # vertex weight: sqrt of the summed scores of its visible faces in this view
        sc = np.asarray(S[vis, k].todense()).ravel()
        wv = np.zeros(nv); np.add.at(wv, fv.ravel(), np.repeat(sc, 3))
        return k, (vids, col, np.sqrt(wv[vids]))
    with ThreadPoolExecutor(n_workers(8)) as ex:
        for k, res in ex.map(view, range(len(cams))):
            if res is None:
                continue
            vids, col, w = res
            good = np.isfinite(col).all(1) & (w > 0)
            bsum[vids[good]] += col[good] * w[good, None]; wsum[vids[good]] += w[good]
            lut = np.full(nv, -1, np.int64); lut[vids] = np.arange(len(vids))
            fk = np.where(label == k)[0]
            if len(fk):
                li = lut[Fi[fk]]
                ok = li >= 0
                c = np.full((len(fk), 3, 3), np.nan)
                c[ok] = col[li[ok]]
                own[fk] = c
    blend = bsum / np.maximum(wsum, 1e-12)[:, None]
    d = blend[Fi] - own                                         # (nf, 3 corners, RGB)
    d[~np.isfinite(d)] = 0
    d[~fl] = 0
    d = np.clip(d, -80, 80)
    log("  multiband: median |low-band correction| %.1f (levels)" % np.median(np.abs(d[fl])))
    # rasterise the corner corrections over the atlas in horizontal bands
    size = atlas.shape[0]
    band = 1024
    fidx = np.where(fl)[0]
    tri = uv_px[fidx]
    ymin, ymax = tri[:, :, 1].min(1), tri[:, :, 1].max(1)
    for y0 in range(0, size, band):
        y1 = min(size, y0 + band)
        m0, m1_ = max(0, y0 - 8), min(size, y1 + 8)              # margin for dilation
        sel = np.where((ymax >= m0 - 1) & (ymin <= m1_ + 1))[0]
        if len(sel) == 0:
            continue
        P = tri[sel].reshape(-1, 2) - [0, m0]
        Fb = np.arange(len(sel) * 3).reshape(-1, 3)
        ids, _, b0, b1 = raster.rasterize(P, np.ones(len(P)), Fb, size, m1_ - m0, max_bbox=1 << 20, bary=True)
        cov = ids >= 0
        corr = np.zeros((m1_ - m0, size, 3), np.float32)
        f = fidx[sel[ids[cov]]]
        l0, l1 = b0[cov][:, None], b1[cov][:, None]
        corr[cov] = l0 * d[f, 0] + l1 * d[f, 1] + (1 - l0 - l1) * d[f, 2]
        # extend into the chart padding: copy from the nearest covered texel
        if cov.any() and (~cov).any():
            _, lab = cv2.distanceTransformWithLabels((~cov).astype(np.uint8), cv2.DIST_L2, 3,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(cov)
            src = np.zeros(lab.max() + 1, np.int64)
            src[lab[ys, xs]] = ys * size + xs
            near = src[lab]
            far = (~cov) & (cv2.distanceTransform((~cov).astype(np.uint8), cv2.DIST_L2, 3) > 6)
            corr = corr.reshape(-1, 3)[near.reshape(-1)].reshape(corr.shape)
            corr[far] = 0
        a = atlas[y0:y1].astype(np.float32)
        atlas[y0:y1] = np.clip(a + corr[y0 - m0:y1 - m0, :, ::-1], 0, 255).astype(np.uint8)
