"""GPU plane-sweep stereo (same algorithm as mvs.compute_depth, PyTorch/CUDA)."""
import numpy as np


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def compute_depth_gpu(ref, nbrs, zmin, zmax, n_planes=128, win=7, best_k=2, mem_budget=3e8):
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda")
    h, w = ref.h, ref.w
    pad = win // 2

    def box(x):
        return F.avg_pool2d(x, win, stride=1, padding=pad, count_include_pad=False)

    R = torch.from_numpy(ref.gray).to(dev)[None, None]
    mu_r = box(R)
    var_r = (box(R * R) - mu_r * mu_r).clamp_min(1e-6)
    Nn = len(nbrs)
    hn, wn = nbrs[0].h, nbrs[0].w
    NB = torch.stack([torch.from_numpy(nb.gray) for nb in nbrs]).to(dev)[:, None]      # (Nn,1,hn,wn)
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    P = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.ones(h * w, device=dev)])     # (3,HW)
    inv = np.linspace(1 / zmin, 1 / zmax, n_planes)
    depths = 1 / inv
    Kinv = ref.Kinv
    rel = []
    for nb in nbrs:
        R_rel = nb.R @ ref.R.T
        t_rel = nb.t - R_rel @ ref.t
        rel.append((nb.K, R_rel, t_rel))
    n = np.array([0, 0, 1.0])
    cost = torch.empty((n_planes, h, w), dtype=torch.float16, device=dev)
    from shapekit.util import fast_mode
    if fast_mode():
        # planes are independent -> batch as many as fit in ~40% of free GPU memory (identical results,
        # far fewer kernel launches than the fixed 300 MB budget)
        free, _ = torch.cuda.mem_get_info()
        mem_budget = max(mem_budget, min(0.4 * free, 3e9))
    B = int(max(1, min(n_planes, mem_budget // (Nn * h * w * 4 * 6))))
    for b0 in range(0, n_planes, B):
        ds = depths[b0:b0 + B]
        Hs = np.stack([K @ (Rr + np.outer(tr, n) / d) @ Kinv for d in ds for (K, Rr, tr) in rel])  # (B*Nn,3,3)
        Ht = torch.from_numpy(Hs).float().to(dev)
        q = Ht @ P                                                     # (B*Nn,3,HW)
        z = q[:, 2]
        u = q[:, 0] / z.clamp_min(1e-9)
        v = q[:, 1] / z.clamp_min(1e-9)
        valid = (z > 0) & (u >= 0) & (u <= wn - 1) & (v >= 0) & (v <= hn - 1)
        grid = torch.stack([u / (wn - 1) * 2 - 1, v / (hn - 1) * 2 - 1], -1).reshape(-1, h, w, 2)
        src = NB.repeat(len(ds), 1, 1, 1)
        S = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # (B*Nn,1,h,w)
        mu_n = box(S)
        var_n = (box(S * S) - mu_n * mu_n).clamp_min(1e-6)
        cov = box(R * S) - mu_r * mu_n
        c = 1 - cov / torch.sqrt(var_r * var_n)
        c = c[:, 0].masked_fill(~valid.reshape(-1, h, w), 2.0).reshape(len(ds), Nn, h, w)
        if Nn > best_k:
            c = torch.topk(c, best_k, dim=1, largest=False).values
        cost[b0:b0 + len(ds)] = c.mean(1).half()
        del q, z, u, v, valid, grid, src, S, mu_n, var_n, cov, c
    costf = cost.float()
    best = torch.argmin(costf, 0)
    cmin = torch.gather(costf, 0, best[None])[0]
    b0_ = best.clamp(1, n_planes - 2)
    c0 = torch.gather(costf, 0, (b0_ - 1)[None])[0]
    c1 = torch.gather(costf, 0, b0_[None])[0]
    c2 = torch.gather(costf, 0, (b0_ + 1)[None])[0]
    denom = c0 - 2 * c1 + c2
    off = torch.where(denom > 1e-6, 0.5 * (c0 - c2) / denom.clamp_min(1e-6), torch.zeros_like(denom)).clamp(-0.5, 0.5)
    invt = torch.from_numpy(inv).float().to(dev)
    inv_d = invt[b0_] + off * float(inv[1] - inv[0])
    depth = (1 / inv_d)
    edge = (best == 0) | (best == n_planes - 1)
    tex = torch.sqrt(var_r[0, 0]) > 0.012
    validref = torch.from_numpy(ref.valid).to(dev)
    mask = (cmin < 0.6) & ~edge & tex & validref
    depth = torch.where(mask, depth, torch.zeros_like(depth))
    out = depth.cpu().numpy().astype(np.float32), (1 - cmin).cpu().numpy().astype(np.float32)
    del cost, costf
    return out
