"""CUDA z-buffer rasteriser (PyTorch). Same contract as raster.rasterize.

Each covered pixel sample gets a 64-bit key = (float32 depth bits << 32) | face_id; a scatter 'amin'
then yields nearest depth and its face in one pass (positive float32 bit patterns sort like the floats).
"""
import numpy as np

_OK = None


def available():
    global _OK
    if _OK is None:
        try:
            import os
            import torch
            _OK = torch.cuda.is_available() and not os.environ.get("SHAPEKIT_NO_GPU")
        except Exception:
            _OK = False
    return _OK


def rasterize_gpu(uv, z, F, w, h, near=1e-6, max_bbox=4096, chunk=6_000_000, bary=False):
    import torch
    dev = torch.device("cuda")
    UV = torch.as_tensor(np.asarray(uv, np.float32), device=dev)
    Zv = torch.as_tensor(np.asarray(z, np.float32), device=dev)
    Ft = torch.as_tensor(np.asarray(F, np.int64), device=dev)
    return rasterize_gpu_t(UV, Zv, Ft, w, h, near, max_bbox, chunk, bary, to_numpy=True)


def rasterize_gpu_t(UV, Zv, Ft, w, h, near=1e-6, max_bbox=4096, chunk=6_000_000, bary=False, to_numpy=False):
    """Tensor version (inputs already on the GPU: UV float32 (nv,2), Zv float32 (nv,), Ft int64 (nf,3))."""
    import torch
    dev = UV.device
    P = UV[Ft]                                   # (nf,3,2)
    Z = Zv[Ft]                                   # (nf,3)
    ok = (Z > near).all(1)
    mn = torch.floor(P.min(1).values).long()
    mx = torch.ceil(P.max(1).values).long()
    ok &= (mx[:, 0] >= 0) & (mx[:, 1] >= 0) & (mn[:, 0] < w) & (mn[:, 1] < h)
    mn[:, 0].clamp_(min=0); mn[:, 1].clamp_(min=0)
    mx[:, 0].clamp_(max=w - 1); mx[:, 1].clamp_(max=h - 1)
    bw, bh = mx[:, 0] - mn[:, 0] + 1, mx[:, 1] - mn[:, 1] + 1
    ok &= (bw > 0) & (bh > 0) & (bw * bh <= max_bbox)
    fids = torch.nonzero(ok)[:, 0]
    INF = torch.iinfo(torch.int64).max
    buf = torch.full((h * w,), INF, dtype=torch.int64, device=dev)
    sz = torch.maximum(bw[fids], bh[fids])
    for lo, hi in [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 4097)]:
        g = fids[(sz > lo) & (sz <= hi)]
        if len(g) == 0:
            continue
        s = hi
        oy, ox = torch.meshgrid(torch.arange(s, device=dev), torch.arange(s, device=dev), indexing="ij")
        ox, oy = ox.reshape(-1), oy.reshape(-1)
        step = max(1, chunk // (s * s))
        for a in range(0, len(g), step):
            f = g[a:a + step]
            px = mn[f, 0][:, None] + ox[None]
            py = mn[f, 1][:, None] + oy[None]
            inside = (px <= mx[f, 0][:, None]) & (py <= mx[f, 1][:, None])
            p0, p1, p2 = P[f, 0], P[f, 1], P[f, 2]
            cx, cy = px.float(), py.float()
            d = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
            d = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)[:, None]
            l0 = ((p1[:, 1] - p2[:, 1])[:, None] * (cx - p2[:, 0][:, None]) +
                  (p2[:, 0] - p1[:, 0])[:, None] * (cy - p2[:, 1][:, None])) / d
            l1 = ((p2[:, 1] - p0[:, 1])[:, None] * (cx - p2[:, 0][:, None]) +
                  (p0[:, 0] - p2[:, 0])[:, None] * (cy - p2[:, 1][:, None])) / d
            l2 = 1 - l0 - l1
            e = -1e-6
            inside &= (l0 >= e) & (l1 >= e) & (l2 >= e)
            zz = l0 * Z[f, 0][:, None] + l1 * Z[f, 1][:, None] + l2 * Z[f, 2][:, None]
            inside &= zz > near
            n_i, k_i = torch.nonzero(inside, as_tuple=True)
            if len(n_i) == 0:
                continue
            pix = py[n_i, k_i] * w + px[n_i, k_i]
            zbits = zz[n_i, k_i].contiguous().view(torch.int32).to(torch.int64)
            key = (zbits << 32) | f[n_i]
            buf.scatter_reduce_(0, pix, key, reduce="amin")
            del px, py, inside, l0, l1, l2, zz, n_i, k_i, pix, key
    valid = buf != INF
    ids = torch.where(valid, buf & 0xFFFFFFFF, torch.full_like(buf, -1)).to(torch.int32)
    zb = torch.full((h * w,), float("inf"), dtype=torch.float32, device=dev)
    zb[valid] = (buf[valid] >> 32).to(torch.int32).view(torch.float32)
    if not to_numpy and not bary:
        return ids.reshape(h, w), zb.reshape(h, w)
    ids_np = ids.reshape(h, w).cpu().numpy()
    zb_np = zb.reshape(h, w).cpu().numpy()
    if not bary:
        return ids_np, zb_np
    # barycentrics at pixel centres for the winning face
    b0 = torch.zeros(h * w, dtype=torch.float32, device=dev)
    b1 = torch.zeros(h * w, dtype=torch.float32, device=dev)
    pv = torch.nonzero(valid)[:, 0]
    f = ids.long()[pv]
    cx = (pv % w).float(); cy = (pv // w).float()
    p0, p1, p2 = P[f, 0], P[f, 1], P[f, 2]
    d = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
    d = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    b0[pv] = ((p1[:, 1] - p2[:, 1]) * (cx - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (cy - p2[:, 1])) / d
    b1[pv] = ((p2[:, 1] - p0[:, 1]) * (cx - p2[:, 0]) + (p0[:, 0] - p2[:, 0]) * (cy - p2[:, 1])) / d
    return ids_np, zb_np, b0.reshape(h, w).cpu().numpy(), b1.reshape(h, w).cpu().numpy()
