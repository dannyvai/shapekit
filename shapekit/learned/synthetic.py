"""Synthetic test scene with exact ground-truth depth (ray-cast textured ground + boxes), used by the
PatchMatch self-test and `shapekit doctor`."""
import numpy as np
import cv2


def synthetic_views(tex_path, W=960, H=540, f=800.0):
    """Ray-cast a textured site (ground plane 20x20 m + three boxes) from 6 low oblique cameras: ground truth
    depth for every pixel; vectorised, no rasteriser needed."""
    from .. import camera as cm  # noqa: F401
    rng = np.random.default_rng(1)
    tex = cv2.resize(cv2.imread(tex_path), (4096, 4096)).astype(np.float32)
    boxes = [(-2, 1, 1.2, 1.0), (2, -1, 1.5, 0.6), (0, 3, 0.8, 1.5)]
    boff = [rng.uniform(0, 0.8, 2) for _ in boxes]
    K = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]])
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    rays_c = np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(K).T          # camera z = 1

    class SV:
        pass
    views = []
    for k in range(6):
        C = np.array([-6 + 0.5 * k, -7.0, 1.6]); look = np.array([0 + 0.3 * k, 1.0, 0.0])
        z = look - C; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
        R = np.stack([x, y, z]); t = -R @ C
        d = rays_c @ R                                    # world directions, scaled so camera depth == ray parameter
        best = np.full((H, W), np.inf); u = np.zeros((H, W)); v = np.zeros((H, W))
        tg = np.where(d[..., 2] < -1e-9, -C[2] / np.minimum(d[..., 2], -1e-9), np.inf)
        P = C + d * tg[..., None]
        ok = np.isfinite(tg) & (np.abs(P[..., 0]) <= 10) & (np.abs(P[..., 1]) <= 10)
        best = np.where(ok, tg, best)
        u = np.where(ok, (P[..., 0] + 10) / 20, u); v = np.where(ok, (P[..., 1] + 10) / 20, v)
        for (cx, cy, s, hgt), o in zip(boxes, boff):
            lo = np.array([cx - s / 2, cy - s / 2, 0.0]); hi = np.array([cx + s / 2, cy + s / 2, hgt])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (lo - C) / d; t2 = (hi - C) / d
            tmin = np.minimum(t1, t2); tmax = np.maximum(t1, t2)
            tn = np.nanmax(tmin, -1); tf = np.nanmin(tmax, -1)
            axis = np.nanargmax(tmin, -1)
            hit = (tn < tf) & (tn > 1e-6) & (tn < best)
            Pb = C + d * tn[..., None]
            rel = (Pb - lo) / (hi - lo)
            a = np.where(axis == 0, rel[..., 1], rel[..., 0]); b = np.where(axis == 2, rel[..., 1], rel[..., 2])
            best = np.where(hit, tn, best)
            u = np.where(hit, o[0] + 0.2 * a, u); v = np.where(hit, o[1] + 0.2 * b, v)
        valid = np.isfinite(best)
        mx = (u * 4095).astype(np.float32); my = (v * 4095).astype(np.float32)
        img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR)
        img = np.where(valid[..., None], img, 0).clip(0, 255).astype(np.uint8)
        sv = SV(); sv.color = img; sv.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        sv.valid = valid; sv.R, sv.t, sv.K = R, t, K; sv.Kinv = np.linalg.inv(K); sv.h, sv.w = H, W; sv.C = C
        sv.gt = np.where(valid, best, 0).astype(np.float32)
        views.append(sv)
    return views


def _shrink(v, s):
    class SV:
        pass
    o = SV()
    o.h, o.w = int(v.h * s), int(v.w * s)
    o.gray = cv2.resize(v.gray, (o.w, o.h), interpolation=cv2.INTER_AREA)
    o.color = cv2.resize(v.color, (o.w, o.h), interpolation=cv2.INTER_AREA)
    o.valid = cv2.resize(v.valid.astype(np.uint8), (o.w, o.h), interpolation=cv2.INTER_NEAREST) > 0
    S = np.diag([s, s, 1.0]); S[0, 2] = S[1, 2] = 0.5 * s - 0.5
    o.K = S @ v.K; o.Kinv = np.linalg.inv(o.K); o.R, o.t, o.C = v.R, v.t, v.C
    o.gt = cv2.resize(v.gt, (o.w, o.h), interpolation=cv2.INTER_NEAREST)
    return o
