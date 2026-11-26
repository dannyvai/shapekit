"""GPU plane-sweep stereo (same algorithm as mvs.compute_depth, PyTorch/CUDA)."""
import numpy as np


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def compute_depth_gpu(ref, nbrs, zmin, zmax, n_planes=128, win=7, best_k=2, mem_budget=3e8, device="cuda"):
    """Memory: the cost of each depth plane is reduced to a running best (value, plane index, and the costs of
    the planes on either side for the sub-plane fit) as soon as it is computed, instead of keeping the whole
    (planes x H x W) cost volume: ~7 values per pixel instead of 3 x n_planes (256 planes of a 27 MP photo
    needed ~42 GB, more than any GPU, and Windows then silently pages GPU memory to system RAM - 100 % GPU,
    no progress). Results are identical (costs are still rounded to float16 per plane, ties keep the first plane)."""
    import torch
    import torch.nn.functional as F
    dev = torch.device(device)
    h, w = ref.h, ref.w
    pad = win // 2

    def box(x):
        return F.avg_pool2d(x, win, stride=1, padding=pad, count_include_pad=False)

    R = torch.from_numpy(ref.gray).to(dev)[None, None]
    mu_r = box(R)
    var_r = (box(R * R) - mu_r * mu_r).clamp_min(1e-6)
    Nn = len(nbrs)
    # neighbours may differ in size (e.g. a portrait and a landscape video): process them in same-size
    # groups; with a single group this is exactly the original batched computation
    groups = {}
    for k_, nb in enumerate(nbrs):
        groups.setdefault((nb.h, nb.w), []).append(k_)
    NBs = {sz: torch.stack([torch.from_numpy(nbrs[k_].gray) for k_ in idxs]).to(dev)[:, None]
           for sz, idxs in groups.items()}
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
    best_c = torch.full((h, w), float("inf"), dtype=torch.float32, device=dev)   # running minimum cost
    best_i = torch.zeros((h, w), dtype=torch.int32, device=dev)                  # ... and its plane
    c_lo = torch.zeros((h, w), dtype=torch.float32, device=dev)                  # cost of plane best-1
    c_hi = torch.zeros((h, w), dtype=torch.float32, device=dev)                  # cost of plane best+1
    c_prev = None
    from ..util import fast_mode
    if fast_mode() and dev.type == "cuda":
        # planes are independent -> batch as many as fit in ~40% of free GPU memory (identical results,
        # far fewer kernel launches than the fixed 300 MB budget)
        free, _ = torch.cuda.mem_get_info()
        mem_budget = max(mem_budget, min(0.4 * free, 3e9))
    B = int(max(1, min(n_planes, mem_budget // (Nn * h * w * 4 * 6))))
    for b0 in range(0, n_planes, B):
        ds = depths[b0:b0 + B]
        c = torch.empty((len(ds), Nn, h, w), dtype=torch.float32, device=dev)
        for (hn, wn), idxs in groups.items():
            Ng = len(idxs)
            Hs = np.stack([rel[k_][0] @ (rel[k_][1] + np.outer(rel[k_][2], n) / d) @ Kinv
                           for d in ds for k_ in idxs])                    # (B*Ng,3,3)
            Ht = torch.from_numpy(Hs).float().to(dev)
            q = Ht @ P                                                     # (B*Ng,3,HW)
            z = q[:, 2]
            u = q[:, 0] / z.clamp_min(1e-9)
            v = q[:, 1] / z.clamp_min(1e-9)
            valid = (z > 0) & (u >= 0) & (u <= wn - 1) & (v >= 0) & (v <= hn - 1)
            grid = torch.stack([u / (wn - 1) * 2 - 1, v / (hn - 1) * 2 - 1], -1).reshape(-1, h, w, 2)
            src = NBs[(hn, wn)].repeat(len(ds), 1, 1, 1)
            S = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # (B*Ng,1,h,w)
            mu_n = box(S)
            var_n = (box(S * S) - mu_n * mu_n).clamp_min(1e-6)
            cov = box(R * S) - mu_r * mu_n
            cg = 1 - cov / torch.sqrt(var_r * var_n)
            cg = cg[:, 0].masked_fill(~valid.reshape(-1, h, w), 2.0).reshape(len(ds), Ng, h, w)
            if len(groups) == 1:
                c = cg
            else:
                c[:, idxs] = cg
            del q, z, u, v, valid, grid, src, S, mu_n, var_n, cov, cg
        if Nn > best_k:
            c = torch.topk(c, best_k, dim=1, largest=False).values
        cm_ = c.mean(1).half().float()                                     # same float16 rounding as before
        del c
        for j in range(cm_.shape[0]):
            k = b0 + j
            ck = cm_[j]
            if c_prev is not None:
                c_hi = torch.where(best_i == k - 1, ck, c_hi)                 # neighbour above the current best
            better = ck < best_c                                           # strict: ties keep the first plane
            best_c = torch.where(better, ck, best_c)
            best_i = torch.where(better, torch.full_like(best_i, k), best_i)
            c_lo = torch.where(better, c_prev if c_prev is not None else ck, c_lo)
            c_prev = ck
        del cm_
    best = best_i.long()
    cmin = best_c
    b0_ = best.clamp(1, n_planes - 2)
    # at edge planes (0, n-1) the fit is irrelevant: those pixels are masked out below
    c0, c1, c2 = c_lo, best_c, c_hi
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
    del best_c, best_i, c_lo, c_hi, c_prev
    return out
