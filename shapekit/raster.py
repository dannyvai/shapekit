"""Vectorised triangle rasteriser (numpy z-buffer) -> face-id / depth / barycentric buffers."""
import numpy as np

from . import camera as cm


def project_vertices(V, rvec, tvec, K):
    R = cm.rodrigues_to_R(rvec)[0]
    Xc = V @ R.T + np.asarray(tvec)
    z = Xc[:, 2]
    zs = np.where(z > 1e-9, z, 1e-9)
    u = Xc[:, 0] / zs * K[0, 0] + K[0, 2]
    v = Xc[:, 1] / zs * K[1, 1] + K[1, 2]
    return np.stack([u, v], 1), z


def rasterize(uv, z, F, w, h, near=1e-6, max_bbox=4096, chunk=2_000_000, bary=False, gpu=None):
    """Returns face_id (h,w) int32 (-1 empty) and depth (h,w) float32 (inf empty) [+ barycentrics].
    Uses the CUDA rasteriser when available (gpu=None -> auto), else the numpy one."""
    if gpu is not False and len(F) > 5000:
        try:
            from .learned.raster_gpu import available, rasterize_gpu
            if available():
                return rasterize_gpu(uv, z, F, w, h, near=near, max_bbox=max_bbox, bary=bary)
        except Exception as e:                       # pragma: no cover - fall back to CPU
            print("GPU rasteriser failed (%s), using CPU" % e)
    return rasterize_cpu(uv, z, F, w, h, near, max_bbox, chunk, bary)


def rasterize_cpu(uv, z, F, w, h, near=1e-6, max_bbox=4096, chunk=2_000_000, bary=False):
    """Numpy reference implementation."""
    P = uv[F]                                  # (nf,3,2)
    Z = z[F]
    ok = (Z > near).all(1)
    mn = np.floor(P.min(1)).astype(np.int64)
    mx = np.ceil(P.max(1)).astype(np.int64)
    ok &= (mx[:, 0] >= 0) & (mx[:, 1] >= 0) & (mn[:, 0] < w) & (mn[:, 1] < h)
    mn = np.maximum(mn, 0); mx = np.minimum(mx, [w - 1, h - 1])
    bw, bh = mx[:, 0] - mn[:, 0] + 1, mx[:, 1] - mn[:, 1] + 1
    ok &= (bw > 0) & (bh > 0) & (bw * bh <= max_bbox)
    fids = np.where(ok)[0]
    zbuf = np.full(h * w, np.inf, np.float64)
    ibuf = np.full(h * w, -1, np.int64)
    b0buf = np.zeros(h * w, np.float32) if bary else None
    b1buf = np.zeros(h * w, np.float32) if bary else None
    # group by bbox size class to vectorise
    sz = np.maximum(bw[fids], bh[fids])
    for lo, hi in [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 4097)]:
        g = fids[(sz > lo) & (sz <= hi)] if lo else fids[sz <= hi]
        if len(g) == 0:
            continue
        s = hi
        per = s * s
        step = max(1, chunk // per)
        oy, ox = np.mgrid[0:s, 0:s]
        ox, oy = ox.ravel(), oy.ravel()
        for a in range(0, len(g), step):
            f = g[a:a + step]
            px = mn[f, 0][:, None] + ox[None]           # (n, per)
            py = mn[f, 1][:, None] + oy[None]
            inside = (px <= mx[f, 0][:, None]) & (py <= mx[f, 1][:, None])
            p0, p1, p2 = P[f, 0], P[f, 1], P[f, 2]
            cx, cy = px + 0.0, py + 0.0
            d = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
            d = np.where(np.abs(d) < 1e-12, 1e-12, d)[:, None]
            l0 = ((p1[:, 1] - p2[:, 1])[:, None] * (cx - p2[:, 0][:, None]) +
                  (p2[:, 0] - p1[:, 0])[:, None] * (cy - p2[:, 1][:, None])) / d
            l1 = ((p2[:, 1] - p0[:, 1])[:, None] * (cx - p2[:, 0][:, None]) +
                  (p0[:, 0] - p2[:, 0])[:, None] * (cy - p2[:, 1][:, None])) / d
            l2 = 1 - l0 - l1
            e = -1e-6
            inside &= (l0 >= e) & (l1 >= e) & (l2 >= e)
            zz = l0 * Z[f, 0][:, None] + l1 * Z[f, 1][:, None] + l2 * Z[f, 2][:, None]
            n_i, k_i = np.nonzero(inside)
            if len(n_i) == 0:
                continue
            pix = py[n_i, k_i] * w + px[n_i, k_i]
            zv = zz[n_i, k_i]
            fv = f[n_i]
            q0, q1 = l0[n_i, k_i], l1[n_i, k_i]
            o = np.lexsort((zv, pix))
            pix, zv, fv, q0, q1 = pix[o], zv[o], fv[o], q0[o], q1[o]
            first = np.concatenate([[True], pix[1:] != pix[:-1]])
            pix, zv, fv, q0, q1 = pix[first], zv[first], fv[first], q0[first], q1[first]
            better = zv < zbuf[pix]
            zbuf[pix[better]] = zv[better]
            ibuf[pix[better]] = fv[better]
            if bary:
                b0buf[pix[better]] = q0[better]; b1buf[pix[better]] = q1[better]
    if bary:
        return (ibuf.reshape(h, w).astype(np.int32), zbuf.reshape(h, w).astype(np.float32),
                b0buf.reshape(h, w), b1buf.reshape(h, w))
    return ibuf.reshape(h, w).astype(np.int32), zbuf.reshape(h, w).astype(np.float32)


def shade(V, F, ids, view_dir_cam=None, R=None):
    """Simple headlight shading image from a face-id buffer."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-20
    out = np.zeros(ids.shape, np.uint8)
    m = ids >= 0
    if R is not None:
        nc = n[ids[m]] @ R.T
        out[m] = (np.abs(nc @ np.array([0.25, -0.35, -0.9]) / 1.0) * 235 + 20).clip(0, 255)
    return out


def render_textured(V, F, UV, face_uv, atlas, rvec, tvec, K, w, h, bg=(0, 0, 0)):
    """Render a textured mesh (atlas BGR uint8, UV in [0,1] with v up) from a pinhole camera."""
    uv, z = project_vertices(V, rvec, tvec, K)
    ids, zb, b0, b1 = rasterize(uv, z, F, w, h, bary=True)
    out = np.zeros((h, w, 3), np.uint8); out[:] = bg
    m = ids >= 0
    f = ids[m]
    l0, l1 = b0[m], b1[m]
    T = UV[face_uv[f]]
    t = l0[:, None] * T[:, 0] + l1[:, None] * T[:, 1] + (1 - l0 - l1)[:, None] * T[:, 2]
    S = atlas.shape[0]
    px = np.clip((t[:, 0] * S).astype(int), 0, S - 1)
    py = np.clip(((1 - t[:, 1]) * S).astype(int), 0, S - 1)
    out[m] = atlas[py, px]
    return out, ids
