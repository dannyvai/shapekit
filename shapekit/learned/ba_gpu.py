"""GPU bundle adjustment (PyTorch, float64). Same model, interface and LM/Schur scheme as ba.bundle_adjust,
but Jacobians come from vmap(jacfwd) and the reduced camera system is assembled + solved on the GPU.

Reduced system: S = U - sum_p sum_{a,b in obs(p)} Y_a W_b^T  with Y = W V^-1, assembled by scattering
per-observation-pair blocks (only pairs sharing a point), in chunks to bound memory.
"""
import numpy as np


def available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _proj_single(r, t, X, I):
    import torch
    th = torch.sqrt((r * r).sum() + 1e-30)
    k = r / th
    c, s = torch.cos(th), torch.sin(th)
    Xc = X * c + torch.linalg.cross(k, X) * s + k * (k * X).sum() * (1 - c) + t
    z = Xc[2]
    x, y = Xc[0] / z, Xc[1] / z
    if I.shape[0] > 8:                       # fisheye model code in I[8] (static shape -> frame path unchanged)
        r0 = torch.sqrt(x * x + y * y + 1e-30)
        th = torch.atan(r0)
        code = I[8]
        fr = torch.where(code > 1.5, 2 * torch.sin(th / 2) / r0, torch.where(code > 0.5, th / r0, torch.ones_like(r0)))
        x, y = x * fr, y * fr
    f, cx, cy, k1, k2, k3, p1, p2 = I[0], I[1], I[2], I[3], I[4], I[5], I[6], I[7]
    r2 = x * x + y * y
    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return torch.stack([f * xd + cx, f * yd + cy])


PRECISION = __import__("os").environ.get("SHAPEKIT_BA_PRECISION", "double")   # double | mixed


def _jac_transform():
    """Forward-mode jacfwd normally. In the frozen (PyInstaller) app forward-mode AD can't start: its first use
    imports torch._decomp.decompositions_for_jvp, which TorchScript-compiles functions and needs Python source
    files that a frozen app doesn't ship. Reverse mode (jacrev) gives the same derivatives without that."""
    import sys
    from torch.func import jacfwd, jacrev
    if getattr(sys, "frozen", False) or __import__("os").environ.get("SHAPEKIT_BA_JACREV"):
        return jacrev
    return jacfwd


def bundle_adjust_gpu(cam_r, cam_t, pts, intr, cam_sensor, obs_cam, obs_pt, obs_uv,
                      fixed_cams=None, intr_mask=None, iters=10, huber=2.0, verbose=False, tol=1e-6,
                      pair_chunk=100_000, obs_chunk=100_000,
                      obs_w=None, prior_idx=None, prior_X=None, prior_info=None, precision=None, progress=None):
    """precision: 'double' (default, everything float64) or 'mixed': Jacobians and Schur-complement blocks in
    float32 (consumer GPUs run fp64 at 1/32-1/64 speed), accumulation, solve, residuals and the LM acceptance
    test in float64 -- steps are slightly less exact, the cost that is minimised is the same."""
    import torch
    from torch.func import vmap, jacfwd
    dev = torch.device("cuda")
    dt = torch.float64
    precision = precision or PRECISION
    jdt = torch.float32 if precision == "mixed" else dt
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
    L = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=dev)

    cr, ct, P, I = T(cam_r).clone(), T(cam_t).clone(), T(pts).clone(), T(intr).clone()
    N, M, S_ = len(cam_r), len(pts), len(intr)
    oc, op, uv = L(obs_cam), L(obs_pt), T(obs_uv)
    csen = L(cam_sensor)
    osen = csen[oc]
    K = len(oc)
    fixed = np.zeros(N, bool) if fixed_cams is None else np.asarray(fixed_cams, bool)
    imask = np.zeros(8, bool) if intr_mask is None else np.asarray(intr_mask, bool)
    free = np.where(~fixed)[0]
    cmap = -np.ones(N, int); cmap[free] = np.arange(len(free))
    ipar = np.where(imask)[0]; nI = len(ipar)
    used_s = np.unique(np.asarray(cam_sensor)[np.asarray(obs_cam)])
    smap = -np.ones(S_, int); smap[used_s] = np.arange(len(used_s))
    C = 6 * len(free) + nI * len(used_s)
    DUMMY = C                                           # fixed-camera columns land here, then dropped
    ncs = 6 + nI
    ci = L(cmap)[oc]
    cam_cols = torch.where(ci[:, None] >= 0, 6 * ci[:, None] + torch.arange(6, device=dev)[None],
                           torch.full((K, 6), DUMMY, device=dev, dtype=torch.long))
    if nI:
        base = 6 * len(free) + nI * L(smap)[osen]
        idx = torch.cat([cam_cols, base[:, None] + torch.arange(nI, device=dev)[None]], 1)
    else:
        idx = cam_cols
    ipar_t = L(ipar)

    # observation pairs sharing a point (incl. a == b)
    order = torch.argsort(op, stable=True)
    counts = torch.bincount(op, minlength=M)
    starts = torch.cumsum(counts, 0) - counts
    n_o = counts[op[order]]                             # track length for each sorted obs
    rep_a = order.repeat_interleave(n_o)                # each obs a repeated n times
    # partner b: the n obs of the same point
    st_a = starts[op[order]].repeat_interleave(n_o)
    within = torch.arange(len(rep_a), device=dev) - torch.repeat_interleave(torch.cumsum(n_o, 0) - n_o, n_o)
    pair_a = rep_a
    pair_b = order[st_a + within]
    npairs = len(pair_a)

    jac = vmap(_jac_transform()(_proj_single, argnums=(0, 1, 2, 3)))
    proj = vmap(_proj_single)

    def residuals(cr, ct, P, I):
        p = proj(cr[oc], ct[oc], P[op], I[osen])
        return p - uv

    def robust(r):
        n = torch.linalg.norm(r, dim=1)
        w = torch.where(n <= huber, torch.ones_like(n), huber / n.clamp_min(1e-12))
        rho = torch.where(n <= huber, n * n, 2 * huber * n - huber * huber)
        return w, rho

    ow = torch.ones(K, dtype=dt, device=dev) if obs_w is None else T(obs_w)
    has_prior = prior_idx is not None and len(prior_idx) > 0
    if has_prior:
        pidx, pX, pinf = L(prior_idx), T(prior_X), T(prior_info)

    def prior_cost(P_):
        if not has_prior:
            return 0.0
        d = P_[pidx] - pX
        return (pinf[:, None] * d * d).sum()

    r = residuals(cr, ct, P, I)
    w, rho = robust(r)
    w = w * ow
    cost = (ow * rho).sum() + prior_cost(P)
    lam = 1e-3
    info = dict(initial_rms=float(torch.sqrt((r ** 2).sum(1).mean())), iterations=0)
    rel = 1.0
    for it in range(iters):
        Jc = torch.empty((K, 2, ncs), dtype=jdt, device=dev)
        Jp = torch.empty((K, 2, 3), dtype=jdt, device=dev)
        sw_all = torch.sqrt(w)
        if jdt == dt:
            crj, ctj, Pj, Ij = cr, ct, P, I
        else:                                   # mixed: linearise in float32
            crj, ctj, Pj, Ij = cr.to(jdt), ct.to(jdt), P.to(jdt), I.to(jdt)
        for a0 in range(0, K, obs_chunk):
            sl = slice(a0, a0 + obs_chunk)
            o_c, o_p, o_s = oc[sl], op[sl], osen[sl]
            Jr, Jt, JX, JI = jac(crj[o_c], ctj[o_c], Pj[o_p], Ij[o_s])
            sw = sw_all[sl][:, None, None].to(jdt)
            Jc[sl] = torch.cat([Jr, Jt] + ([JI[:, :, ipar_t]] if nI else []), 2) * sw
            Jp[sl] = JX * sw
            del Jr, Jt, JX, JI
        rw = (r * sw_all[:, None]).to(jdt)
        torch.cuda.empty_cache()
        U = torch.zeros((C + 1) * (C + 1), dtype=dt, device=dev)
        for a0 in range(0, K, obs_chunk):
            sl = slice(a0, a0 + obs_chunk)
            lin = idx[sl][:, :, None] * (C + 1) + idx[sl][:, None, :]
            U.index_add_(0, lin.reshape(-1), (Jc[sl].transpose(1, 2) @ Jc[sl]).reshape(-1).to(dt))
        U = U.view(C + 1, C + 1)
        gc = torch.zeros(C + 1, dtype=dt, device=dev)
        gc.index_add_(0, idx.reshape(-1), (Jc.transpose(1, 2) @ rw[:, :, None]).reshape(-1).to(dt))
        V = torch.zeros((M, 3, 3), dtype=dt, device=dev)
        V.index_add_(0, op, (Jp.transpose(1, 2) @ Jp).to(dt))
        gp = torch.zeros((M, 3), dtype=dt, device=dev)
        gp.index_add_(0, op, (Jp.transpose(1, 2) @ rw[:, :, None])[:, :, 0].to(dt))
        if has_prior:
            gp.index_add_(0, pidx, pinf[:, None] * (P[pidx] - pX))
            V.index_add_(0, pidx, pinf[:, None, None] * torch.eye(3, dtype=dt, device=dev)[None])
        # W_o = Jc_o^T Jp_o and Y_o = W_o V^-1 are formed on the fly per chunk (memory: only Jc, Jp kept)
        Wof = lambda s: Jc[s].transpose(1, 2) @ Jp[s]
        Udiag = torch.diagonal(U)[:C].clone()
        accepted = False
        for _ in range(8):
            Vd = V + lam * torch.diag_embed(torch.diagonal(V, dim1=1, dim2=2)) + 1e-9 * torch.eye(3, device=dev, dtype=dt)
            Vinv = torch.linalg.inv(Vd)
            Vinv_j = Vinv.to(jdt)
            Smat = U.clone()
            Smat[torch.arange(C), torch.arange(C)] += lam * Udiag + 1e-9
            Sflat = Smat.view(-1)
            for a0 in range(0, npairs, pair_chunk):
                a = pair_a[a0:a0 + pair_chunk]; b = pair_b[a0:a0 + pair_chunk]
                blk = (Wof(a) @ Vinv_j[op[a]]) @ Wof(b).transpose(1, 2)    # (n,ncs,ncs)
                lin = idx[a][:, :, None] * (C + 1) + idx[b][:, None, :]
                Sflat.index_add_(0, lin.reshape(-1), -blk.reshape(-1).to(dt))
                del blk, lin
            rhs = -gc.clone()
            for a0 in range(0, K, obs_chunk):
                sl = slice(a0, a0 + obs_chunk)
                ygp = ((Wof(sl) @ Vinv_j[op[sl]]) @ gp[op[sl]].to(jdt)[:, :, None])[:, :, 0]
                rhs.index_add_(0, idx[sl].reshape(-1), ygp.reshape(-1).to(dt))
            Sc, rc = Smat[:C, :C], rhs[:C]
            Sc = 0.5 * (Sc + Sc.T)
            Lc, info_ = torch.linalg.cholesky_ex(Sc)
            if int(info_) == 0:
                dc = torch.cholesky_solve(rc[:, None], Lc)[:, 0]
            else:
                dc = torch.linalg.lstsq(Sc, rc[:, None]).solution[:, 0]
            dce = torch.cat([dc, torch.zeros(1, dtype=dt, device=dev)]).to(jdt)
            acc = gp.clone()
            for a0 in range(0, K, obs_chunk):
                sl = slice(a0, a0 + obs_chunk)
                acc.index_add_(0, op[sl], (Wof(sl).transpose(1, 2) @ dce[idx[sl]][:, :, None])[:, :, 0].to(dt))
            dp = -(Vinv @ acc[:, :, None])[:, :, 0]
            ncr, nct, nI_ = cr.clone(), ct.clone(), I.clone()
            if len(free):
                d = dc[:6 * len(free)].reshape(-1, 6)
                ft = L(free)
                ncr[ft] += d[:, :3]; nct[ft] += d[:, 3:]
            if nI:
                d = dc[6 * len(free):].reshape(-1, nI)
                for si, s in enumerate(used_s):
                    nI_[int(s), ipar_t] += d[si]
            nP = P + dp
            r2 = residuals(ncr, nct, nP, nI_)
            w2, rho2 = robust(r2)
            w2 = w2 * ow
            c2 = (ow * rho2).sum() + prior_cost(nP)
            if torch.isfinite(c2) and c2 < cost:
                rel = float((cost - c2) / cost)
                cr, ct, P, I, r, w, cost = ncr, nct, nP, nI_, r2, w2, c2
                lam = max(lam / 3, 1e-8)
                accepted = True
                break
            lam *= 6
        info["iterations"] = it + 1
        if progress:
            from ..util import log
            log("    %s: iteration %d/%d, rms %.3f px" % (progress, it + 1, iters, float(torch.sqrt((r ** 2).sum(1).mean()))))
        if verbose:
            print("  GPU BA it %d cost %.4g rms %.3f lam %.1e" % (it, float(cost), float(torch.sqrt((r ** 2).sum(1).mean())), lam))
        if not accepted or rel < tol:
            break
    # depths for diagnostics
    from .. import camera as cm
    out_r, out_t, out_P, out_I = (x.cpu().numpy() for x in (cr, ct, P, I))
    _, z = cm.project(out_r[np.asarray(obs_cam)], out_t[np.asarray(obs_cam)], out_P[np.asarray(obs_pt)],
                      out_I[np.asarray(cam_sensor)[np.asarray(obs_cam)]])
    res = torch.linalg.norm(r, dim=1).cpu().numpy()
    info.update(final_rms=float(np.sqrt((res ** 2).mean())), residuals=res, depths=z, pairs=int(npairs))
    return out_r, out_t, out_P, out_I, info
