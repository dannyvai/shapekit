"""PatchMatch multi-view stereo with per-pixel slanted planes (normals), ACMH-style.

Every reference pixel carries a plane  n . X + d = 0  in camera coordinates (unit normal n facing the camera,
d > 0); its depth is z = -d / (n . K^-1 [u v 1]).  A hypothesis is scored by warping a sparse window (9x9,
stride 2 -> 25 samples) through the plane-induced homography into each neighbour view and computing
bilateral-weighted NCC; views are aggregated with the mean of the best k (occlusion robust, like the plane sweep).

Search: random init, red-black checkerboard propagation (8 neighbours at distance 1 and 3), random
refinement with shrinking perturbations; coarse-to-fine (half resolution first, planes are resolution
independent so they upsample directly).  Slanted surfaces (ground seen at grazing angles, walls, steps) are
reconstructed without the stair-casing of fronto-parallel sweeps.

The algorithm is written once against a tiny array backend so it runs with PyTorch (CUDA or CPU) or NumPy
(reference / tests).
"""
import numpy as np


# ----------------------------------------------------------------------------- backends
class _NumpyB:
    name = "numpy"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def f(self, a):
        return np.asarray(a, np.float32)

    def i(self, a):
        return np.asarray(a, np.int64)

    def rand(self, *shape):
        return self.rng.random(shape, dtype=np.float32)

    def randn(self, *shape):
        return self.rng.standard_normal(shape, dtype=np.float32)

    zeros = staticmethod(lambda *s: np.zeros(s, np.float32))
    where = staticmethod(np.where)
    sqrt = staticmethod(np.sqrt)
    exp = staticmethod(np.exp)
    stack = staticmethod(lambda xs, ax: np.stack(xs, ax))

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def argmin0(self, x):
        return np.argmin(x, 0)

    def take0(self, x, idx):
        """x (C, M, ...) , idx (M,) -> x[idx[m], m, ...]"""
        return x[idx, np.arange(x.shape[1])]

    def best_k_mean(self, c, k):
        """c (V, M) -> mean of the k smallest along views."""
        if c.shape[0] <= k:
            return c.mean(0)
        return np.partition(c, k - 1, axis=0)[:k].mean(0)

    def sample(self, img, x, y):
        """bilinear; img (h,w) float32; x,y (...) -> vals, valid"""
        h, w = img.shape
        valid = (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
        xc = np.clip(x, 0, w - 1.001); yc = np.clip(y, 0, h - 1.001)
        x0 = np.floor(xc).astype(np.int64); y0 = np.floor(yc).astype(np.int64)
        fx = (xc - x0).astype(np.float32); fy = (yc - y0).astype(np.float32)
        v = ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy) +
             (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)
        return v, valid

    def to_np(self, a):
        return np.asarray(a)

    def resize_half(self, img):
        import cv2
        return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)


class _TorchB:
    name = "torch"

    def __init__(self, device, seed=0):
        import torch
        import torch.nn.functional as F
        self.t, self.F, self.dev = torch, F, device
        self.gen = torch.Generator(device=device); self.gen.manual_seed(seed)

    def f(self, a):
        return self.t.as_tensor(np.asarray(a, np.float32) if not self.t.is_tensor(a) else a,
                                dtype=self.t.float32, device=self.dev)

    def i(self, a):
        return self.t.as_tensor(a, dtype=self.t.int64, device=self.dev)

    def rand(self, *shape):
        return self.t.rand(shape, generator=self.gen, device=self.dev)

    def randn(self, *shape):
        return self.t.randn(shape, generator=self.gen, device=self.dev)

    def zeros(self, *s):
        return self.t.zeros(s, device=self.dev)

    def where(self, c, a, b):
        t = self.t
        a = a if t.is_tensor(a) else t.tensor(a, device=self.dev, dtype=t.float32)
        b = b if t.is_tensor(b) else t.tensor(b, device=self.dev, dtype=t.float32)
        return t.where(c, a, b)

    def sqrt(self, x):
        return self.t.sqrt(x)

    def exp(self, x):
        return self.t.exp(x)

    def stack(self, xs, ax):
        return self.t.stack(xs, ax)

    def clip(self, x, lo, hi):
        return self.t.clamp(x, lo, hi)

    def argmin0(self, x):
        return self.t.argmin(x, 0)

    def take0(self, x, idx):
        return x[idx, self.t.arange(x.shape[1], device=self.dev)]

    def best_k_mean(self, c, k):
        if c.shape[0] <= k:
            return c.mean(0)
        return self.t.topk(c, k, dim=0, largest=False).values.mean(0)

    def sample(self, img, x, y):
        h, w = img.shape
        valid = (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
        shp = x.shape
        g = self.t.stack([x.reshape(1, -1, 1) / (w - 1) * 2 - 1, y.reshape(1, -1, 1) / (h - 1) * 2 - 1], -1)
        v = self.F.grid_sample(img[None, None], g, mode="bilinear", padding_mode="border", align_corners=True)
        return v.reshape(shp), valid

    def to_np(self, a):
        return a.detach().cpu().numpy()

    def resize_half(self, img):
        return self.F.avg_pool2d(img[None, None], 2)[0, 0]


def backend(prefer="auto", seed=0):
    if prefer in ("auto", "torch", "cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return _TorchB(torch.device("cuda"), seed)
            if prefer in ("torch",):
                return _TorchB(torch.device("cpu"), seed)
        except Exception:
            pass
    return _NumpyB(seed)


# ----------------------------------------------------------------------------- core
class _Level:
    """One resolution level: reference image, rays, window samples and neighbour warps."""

    def __init__(self, B, ref_gray, K, nbrs, win, step, sigma_c=0.12):
        self.B = B
        h, w = ref_gray.shape
        self.h, self.w = h, w
        self.ref = B.f(ref_gray)
        r = win // 2
        o = np.arange(-r, r + 1, step, dtype=np.float32)
        oy, ox = np.meshgrid(o, o, indexing="ij")
        self.ox, self.oy = ox.ravel(), oy.ravel()                    # (S,)
        S = len(self.ox)
        Kinv = np.linalg.inv(K)
        self.Kinv = Kinv
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        self.px, self.py = B.f(xs.ravel()), B.f(ys.ravel())
        rays = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w, np.float32)], 1) @ Kinv.T.astype(np.float32)
        self.ray = B.f(rays)                                           # (HW,3)
        # ref window samples + bilateral weights (per pixel, fixed)
        X = self.px[:, None] + B.f(self.ox)[None]
        Y = self.py[:, None] + B.f(self.oy)[None]
        rv, _ = B.sample(self.ref, X, Y)                               # (HW,S)
        c = rv[:, S // 2:S // 2 + 1]                                   # centre sample == pixel value
        wgt = B.exp(-((rv - c) ** 2) / (2 * sigma_c ** 2) - B.f((self.ox ** 2 + self.oy ** 2) / (2 * (r + 1) ** 2))[None])
        self.wt = wgt / wgt.sum(1)[:, None]
        mu = (self.wt * rv).sum(1)[:, None]
        self.rc = rv - mu
        self.var_r = (self.wt * self.rc * self.rc).sum(1)
        # ray offsets of the window samples: K^-1 [ox, oy, 0]
        self.dray = B.f(np.stack([self.ox * Kinv[0, 0], self.oy * Kinv[1, 1], np.zeros(S, np.float32)], 1))  # (S,3)
        self.nb = []
        for img, Kn, R_rel, t_rel in nbrs:
            A = (Kn @ R_rel @ np.eye(3)).astype(np.float32)            # applied to rays (K_r^-1 q already)
            b = (-(Kn @ t_rel)).astype(np.float32)
            self.nb.append((B.f(img), B.f(A), B.f(b)))

    def cost(self, idx, n, d, best_k=2):
        """idx (M,) pixel indices; n (M,3); d (M,) -> aggregated cost (M,), also invalid -> 2."""
        B = self.B
        ray = self.ray[idx]                                            # (M,3)
        q = ray[:, None, :] + self.dray[None]                          # (M,S,3) sample rays
        c = (q * n[:, None, :]).sum(-1) / d[:, None]                   # (M,S) = n.q / d
        per = []
        for img, A, b in self.nb:
            Aq = q @ A.T                                               # (M,S,3)
            p = Aq + b[None, None, :] * c[..., None]
            z = p[..., 2]
            zs = B.where(z > 1e-6, z, 1e-6)
            u, v = p[..., 0] / zs, p[..., 1] / zs
            s, valid = B.sample(img, u, v)
            valid = valid & (z > 1e-6)
            wv = self.wt[idx] * valid
            mu = (wv * s).sum(1) / B.clip(wv.sum(1), 1e-6, 1e9)
            sc = s - mu[:, None]
            cov = (wv * self.rc[idx] * sc).sum(1)
            var_s = (wv * sc * sc).sum(1)
            ncc = cov / B.sqrt(B.clip(self.var_r[idx] * var_s, 1e-10, 1e9))
            frac = valid.sum(1) / float(valid.shape[1])
            cst = B.where(frac > 0.5, B.clip(1 - ncc, 0, 2), 2.0)
            per.append(cst)
        return B.best_k_mean(B.stack(per, 0), best_k)

    def depth(self, idx, n, d):
        return -d / B_safe(self.B, (n * self.ray[idx]).sum(1))


def B_safe(B, x):
    return B.where(x.__abs__() > 1e-9, x, -1e-9)


def _random_planes(B, ray, m, inv_lo, inv_hi):
    """Random planes for M pixels: inverse depth uniform, normal random in the hemisphere facing the camera."""
    invd = inv_lo + (inv_hi - inv_lo) * B.rand(m)
    z = 1 / invd
    n = B.randn(m, 3)
    n = n / B.sqrt((n * n).sum(1)[:, None] + 1e-12)
    facing = (n * ray).sum(1)
    n = B.where((facing > 0)[:, None], -n, n)
    # keep normals within ~80 deg of the viewing ray (grazing planes are useless)
    rn = ray / B.sqrt((ray * ray).sum(1)[:, None])
    cosv = -(n * rn).sum(1)
    n = B.where((cosv < 0.17)[:, None], -rn, n)
    d = -z * (n * ray).sum(1)
    return n, d


def _perturb(B, n, d, ray, inv_lo, inv_hi, scale):
    m = n.shape[0]
    z = -d / B_safe(B, (n * ray).sum(1))
    invz = 1 / B.clip(z, 1e-6, 1e9)
    invz = B.clip(invz + (B.rand(m) - 0.5) * scale * (inv_hi - inv_lo), min(inv_lo, inv_hi), max(inv_lo, inv_hi))
    z2 = 1 / invz
    n2 = n + B.randn(m, 3) * (scale * 0.5)
    n2 = n2 / B.sqrt((n2 * n2).sum(1)[:, None] + 1e-12)
    rn = ray / B.sqrt((ray * ray).sum(1)[:, None])
    bad = -(n2 * rn).sum(1) < 0.17
    n2 = B.where(bad[:, None], n, n2)
    return n2, z2


def _planes_from(B, n, z, ray):
    return n, -z * (n * ray).sum(1)


NEIGH = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, -3), (0, 3), (-3, 0), (3, 0)]


def _run_level(L, n, d, cost, zmin, zmax, iters, best_k, refine_scale=1.0):
    B = L.B
    h, w = L.h, L.w
    inv_lo, inv_hi = 1 / zmax, 1 / zmin
    ys, xs = np.mgrid[0:h, 0:w]
    color = ((ys + xs) % 2).ravel()
    for it in range(iters):
        for col in (0, 1):
            idx_np = np.flatnonzero(color == col)
            idx = B.i(idx_np)
            ray = L.ray[idx]
            yy, xx = idx_np // w, idx_np % w
            # --- propagation: planes of the 8 checkerboard neighbours
            best_n, best_d, best_c = n[idx], d[idx], cost[idx]
            for dy, dx in NEIGH:
                j = B.i(np.clip(yy + dy, 0, h - 1) * w + np.clip(xx + dx, 0, w - 1))
                cn, cd = n[j], d[j]
                z = -cd / B_safe(B, (cn * ray).sum(1))
                okz = (z >= zmin) & (z <= zmax)
                c = L.cost(idx, cn, B.where(okz, cd, best_d), best_k)
                better = okz & (c < best_c)
                best_n = B.where(better[:, None], cn, best_n)
                best_d = B.where(better, cd, best_d)
                best_c = B.where(better, c, best_c)
            # --- refinement: perturbed / random / mixed hypotheses
            sc = refine_scale * 0.5 ** it
            m = len(idx_np)
            rn_, rd_ = _random_planes(B, ray, m, inv_lo, inv_hi)
            pn, pz = _perturb(B, best_n, best_d, ray, inv_lo, inv_hi, sc)
            z_cur = -best_d / B_safe(B, (best_n * ray).sum(1))
            z_rnd = -rd_ / B_safe(B, (rn_ * ray).sum(1))
            cands = [(pn, pz), (best_n, pz), (pn, z_cur), (rn_, z_cur), (best_n, z_rnd)]
            for cn, cz in cands:
                cn, cd = _planes_from(B, cn, cz, ray)
                okz = (cz >= zmin) & (cz <= zmax) & (cd > 0)
                c = L.cost(idx, cn, B.where(okz, cd, best_d), best_k)
                better = okz & (c < best_c)
                best_n = B.where(better[:, None], cn, best_n)
                best_d = B.where(better, cd, best_d)
                best_c = B.where(better, c, best_c)
            n[idx], d[idx], cost[idx] = best_n, best_d, best_c
    return n, d, cost


def patchmatch_depth(ref, nbrs, zmin, zmax, win=9, step=2, iters=(3, 3), best_k=2, prefer="auto", seed=0,
                     return_normals=False):
    """ref / nbrs: mvs.View objects (undistorted pinhole). Returns depth (h,w) float32 (0 = invalid) and
    confidence (h,w) [, normals (h,w,3) in camera coords]. iters = (coarse level, full level)."""
    import cv2
    B = backend(prefer, seed)
    h, w = ref.h, ref.w

    def rel(nb):
        R_rel = nb.R @ ref.R.T
        return R_rel, nb.t - R_rel @ ref.t
    levels = []
    # coarse level (half resolution): K scaled by 0.5 in pixel-centre convention
    S = np.diag([0.5, 0.5, 1.0]); S[0, 2] = S[1, 2] = -0.25
    Kc = S @ ref.K
    ref_c = cv2.resize(ref.gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    nb_c = []
    for nb in nbrs:
        R_rel, t_rel = rel(nb)
        g = cv2.resize(nb.gray, (nb.w // 2, nb.h // 2), interpolation=cv2.INTER_AREA)
        nb_c.append((g, S @ nb.K, R_rel, t_rel))
    if iters[0] > 0:
        Lc = _Level(B, ref_c, Kc, nb_c, win, step)
        n, d = _random_planes(B, Lc.ray, Lc.h * Lc.w, 1 / zmax, 1 / zmin)
        cost = Lc.cost(B.i(np.arange(Lc.h * Lc.w)), n, d, best_k)
        n, d, cost = _run_level(Lc, n, d, cost, zmin, zmax, iters[0], best_k)
        # upsample planes to full resolution (nearest; planes are resolution independent)
        ys, xs = np.mgrid[0:h, 0:w]
        src = B.i(np.minimum(ys // 2, Lc.h - 1).ravel() * Lc.w + np.minimum(xs // 2, Lc.w - 1).ravel())
        n, d = n[src], d[src]
        del Lc
    nb_f = []
    for nb in nbrs:
        R_rel, t_rel = rel(nb)
        nb_f.append((nb.gray, nb.K, R_rel, t_rel))
    L = _Level(B, ref.gray, ref.K, nb_f, win, step)
    if iters[0] <= 0:
        n, d = _random_planes(B, L.ray, h * w, 1 / zmax, 1 / zmin)
    allidx = B.i(np.arange(h * w))
    # full-level costs of the upsampled planes, guard planes whose depth left the range
    z = -d / B_safe(B, (n * L.ray).sum(1))
    okz = (z >= zmin) & (z <= zmax) & (d > 0)
    rn_, rd_ = _random_planes(B, L.ray, h * w, 1 / zmax, 1 / zmin)
    n = B.where(okz[:, None], n, rn_); d = B.where(okz, d, rd_)
    cost = B.f(np.zeros(h * w, np.float32))
    for a0 in range(0, h * w, 1 << 18):
        sl = allidx[a0:a0 + (1 << 18)]
        cost[a0:a0 + (1 << 18)] = L.cost(sl, n[sl], d[sl], best_k)
    n, d, cost = _run_level(L, n, d, cost, zmin, zmax, iters[1], best_k, refine_scale=0.25 if iters[0] > 0 else 1.0)
    z = -d / B_safe(B, (n * L.ray).sum(1))
    depth = B.to_np(z).reshape(h, w).astype(np.float32)
    cmin = B.to_np(cost).reshape(h, w).astype(np.float32)
    nrm = B.to_np(n).reshape(h, w, 3).astype(np.float32)
    # same validity rules as the plane sweep: matching cost, texture, inside the undistorted image, in range
    mu = cv2.blur(ref.gray, (win, win)); var = cv2.blur(ref.gray * ref.gray, (win, win)) - mu * mu
    tex = np.sqrt(np.maximum(var, 0)) > 0.012
    mask = (cmin < 0.6) & tex & ref.valid & (depth >= zmin) & (depth <= zmax)
    depth = np.where(mask, depth, 0).astype(np.float32)
    out = (depth, (1 - np.clip(cmin, 0, 1)).astype(np.float32))
    return out + (nrm,) if return_normals else out
