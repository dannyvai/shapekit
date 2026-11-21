"""Sparse Levenberg-Marquardt bundle adjustment with Schur complement (points eliminated).

Optimises camera poses, 3D points and (optionally) per-sensor intrinsics
f, cx, cy, k1, k2, k3, p1, p2 -- the equivalent of Metashape's Optimize Cameras.
"""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

from . import camera as cm


def _residuals(cr, ct, P, I, oc, op, osen, uv):
    proj, z = cm.project(cr[oc], ct[oc], P[op], I[osen])
    return proj - uv, z


def _robust(r, delta):
    n = np.linalg.norm(r, axis=1)
    w = np.where(n <= delta, 1.0, delta / np.maximum(n, 1e-12))
    rho = np.where(n <= delta, n * n, 2 * delta * n - delta * delta)
    return w, rho


def _prior_terms(P, prior_idx, prior_X, prior_info):
    """GCP priors on points: cost sum info * |X - X0|^2 ; returns (cost, grad (n,3), info (n,))."""
    if prior_idx is None or len(prior_idx) == 0:
        return 0.0, None, None
    d = P[prior_idx] - prior_X
    return float((prior_info[:, None] * d * d).sum()), prior_info[:, None] * d, prior_info


def bundle_adjust(cam_r, cam_t, pts, intr, cam_sensor, obs_cam, obs_pt, obs_uv,
                  fixed_cams=None, intr_mask=None, iters=10, huber=2.0, verbose=False, tol=1e-6,
                  obs_w=None, prior_idx=None, prior_X=None, prior_info=None, progress=None):
    """obs_w: per-observation weight (e.g. 4 for markers = 0.5 px vs 1 px tie points).
    prior_idx / prior_X / prior_info: point priors (GCPs) with isotropic information 1/sigma^2 (internal units)."""
    """All arrays are modified copies; returns (cam_r, cam_t, pts, intr, info).

    cam_r, cam_t: (N,3); pts: (M,3); intr: (S,8); cam_sensor: (N,) sensor index
    obs_*: per observation; intr_mask: (8,) bool -> which intrinsics to refine (shared per sensor)
    """
    cam_r, cam_t, pts, intr = cam_r.copy(), cam_t.copy(), pts.copy(), intr.copy()
    N, M, S = len(cam_r), len(pts), len(intr)
    oc, op = np.asarray(obs_cam), np.asarray(obs_pt)
    osen = np.asarray(cam_sensor)[oc]
    K = len(oc)
    fixed = np.zeros(N, bool) if fixed_cams is None else np.asarray(fixed_cams, bool)
    imask = np.zeros(8, bool) if intr_mask is None else np.asarray(intr_mask, bool)
    free_cam = np.where(~fixed)[0]
    cam_idx = -np.ones(N, int); cam_idx[free_cam] = np.arange(len(free_cam))
    ipar = np.where(imask)[0]
    nI = len(ipar)
    used_sensors = np.unique(osen)
    sen_idx = -np.ones(S, int); sen_idx[used_sensors] = np.arange(len(used_sensors))
    C = 6 * len(free_cam) + nI * len(used_sensors)
    scale = float(np.median(np.abs(cam_t))) + 1e-3

    ow = np.ones(len(oc)) if obs_w is None else np.asarray(obs_w, float)
    if prior_idx is not None:
        prior_idx = np.asarray(prior_idx, int); prior_X = np.asarray(prior_X, float)
        prior_info = np.asarray(prior_info, float)
    r, z = _residuals(cam_r, cam_t, pts, intr, oc, op, osen, obs_uv)
    w, rho = _robust(r, huber)
    w = w * ow
    cost = (ow * rho).sum() + _prior_terms(pts, prior_idx, prior_X, prior_info)[0]
    lam = 1e-3
    info = dict(initial_rms=float(np.sqrt((r ** 2).sum(1).mean())), iterations=0)

    for it in range(iters):
        # ---------------- Jacobians (central differences, vectorised over observations)
        base = (cam_r[oc], cam_t[oc], pts[op], intr[osen])
        Jc = np.zeros((K, 2, 6)); Jp = np.zeros((K, 2, 3)); Ji = np.zeros((K, 2, nI))

        def f(rv, tv, X, I):
            return cm.project(rv, tv, X, I)[0]

        for j in range(6):
            e = 1e-6 if j < 3 else 1e-6 * scale
            a = [b.copy() for b in base]
            bb = [b.copy() for b in base]
            if j < 3:
                a[0][:, j] += e; bb[0][:, j] -= e
            else:
                a[1][:, j - 3] += e; bb[1][:, j - 3] -= e
            Jc[:, :, j] = (f(*a) - f(*bb)) / (2 * e)
        for j in range(3):
            e = 1e-6 * scale
            a = list(base); bb = list(base)
            a[2] = base[2].copy(); a[2][:, j] += e
            bb[2] = base[2].copy(); bb[2][:, j] -= e
            Jp[:, :, j] = (f(*a) - f(*bb)) / (2 * e)
        for jj, j in enumerate(ipar):
            e = 1e-4 if j < 3 else 1e-6
            a = list(base); bb = list(base)
            a[3] = base[3].copy(); a[3][:, j] += e
            bb[3] = base[3].copy(); bb[3][:, j] -= e
            Ji[:, :, jj] = (f(*a) - f(*bb)) / (2 * e)

        wr = (w[:, None] * r)                         # (K,2)
        sw = np.sqrt(w)[:, None, None]
        Jc_w, Jp_w, Ji_w = Jc * sw, Jp * sw, Ji * sw   # sqrt-weighted
        rw = r * np.sqrt(w)[:, None]

        # camera-side sparse Jacobian (2K x C)
        ci = cam_idx[oc]
        rr, cc, vv = [], [], []
        m = ci >= 0
        if m.any():
            kk = np.where(m)[0]
            rr.append(np.repeat((2 * kk[:, None] + np.arange(2)[None]).ravel(), 6))
            cc.append(np.broadcast_to(6 * ci[kk][:, None, None] + np.arange(6)[None, None, :], (len(kk), 2, 6)).ravel())
            vv.append(Jc_w[kk].reshape(-1))
        if nI:
            kk = np.arange(K)
            base_col = 6 * len(free_cam) + nI * sen_idx[osen]
            rr.append(np.repeat((2 * kk[:, None] + np.arange(2)[None]).ravel(), nI))
            cc.append((np.repeat(base_col, 2)[:, None] + np.arange(nI)[None]).ravel())
            vv.append(Ji_w.reshape(-1))
        if rr:
            JC = sp.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(2 * K, C))
        else:
            JC = sp.csr_matrix((2 * K, C))
        JP = sp.csr_matrix((Jp_w.reshape(-1), (np.repeat(np.arange(2 * K), 3),
                                                (np.repeat(3 * op, 2)[:, None] + np.arange(3)[None]).ravel())),
                           shape=(2 * K, 3 * M))
        rvec = rw.reshape(-1)
        U = (JC.T @ JC).toarray()
        gc = JC.T @ rvec
        V = np.zeros((M, 3, 3))
        JtJ = np.einsum("kai,kaj->kij", Jp_w, Jp_w)
        for a in range(3):
            for b in range(3):
                V[:, a, b] = np.bincount(op, JtJ[:, a, b], minlength=M)
        gp = np.zeros((M, 3))
        g_obs = np.einsum("kai,ka->ki", Jp_w, rw)
        for a in range(3):
            gp[:, a] = np.bincount(op, g_obs[:, a], minlength=M)
        _, pg, pinfo = _prior_terms(pts, prior_idx, prior_X, prior_info)
        if pg is not None:
            np.add.at(gp, prior_idx, pg)
            for a in range(3):
                np.add.at(V[:, a, a], prior_idx, pinfo)
        Wcp = (JC.T @ JP).tocsr()

        accepted = False
        for _ in range(8):
            Ud = U + lam * np.diag(np.diag(U)) + 1e-9 * np.eye(C)
            Vd = V + lam * V * np.eye(3)[None] + 1e-9 * np.eye(3)[None]
            Vinv = np.linalg.inv(Vd)
            Vi = _block_diag(Vinv)
            WV = Wcp @ Vi
            Smat = Ud - (WV @ Wcp.T).toarray()
            rhs = -(gc - WV @ gp.reshape(-1))
            try:
                dc = sla.cho_solve(sla.cho_factor(Smat, check_finite=False), rhs, check_finite=False) if C else np.zeros(0)
            except Exception:
                dc = np.linalg.lstsq(Smat, rhs, rcond=None)[0]
            dp = -(Vi @ (gp.reshape(-1) + Wcp.T @ dc)).reshape(M, 3)
            nr, nt, ni = cam_r.copy(), cam_t.copy(), intr.copy()
            if len(free_cam):
                d = dc[:6 * len(free_cam)].reshape(-1, 6)
                nr[free_cam] += d[:, :3]; nt[free_cam] += d[:, 3:]
            if nI:
                d = dc[6 * len(free_cam):].reshape(-1, nI)
                for s_i, s in enumerate(used_sensors):
                    ni[s, ipar] += d[s_i]
            npnt = pts + dp
            r2, z2 = _residuals(nr, nt, npnt, ni, oc, op, osen, obs_uv)
            w2, rho2 = _robust(r2, huber)
            w2 = w2 * ow
            c2 = (ow * rho2).sum() + _prior_terms(npnt, prior_idx, prior_X, prior_info)[0]
            if np.isfinite(c2) and c2 < cost:
                rel = (cost - c2) / cost
                cam_r, cam_t, intr, pts, r, z, w, cost = nr, nt, ni, npnt, r2, z2, w2, c2
                lam = max(lam / 3, 1e-8)
                accepted = True
                break
            lam *= 6
        info["iterations"] = it + 1
        if progress:
            from .util import log
            log("    %s: iteration %d/%d, rms %.3f px" % (progress, it + 1, iters, np.sqrt((r ** 2).sum(1).mean())))
        if verbose:
            print("  BA it %d cost %.4g rms %.3f lam %.1e" % (it, cost, np.sqrt((r ** 2).sum(1).mean()), lam))
        if not accepted or rel < tol:
            break
    res = np.linalg.norm(r, axis=1)
    info.update(final_rms=float(np.sqrt((res ** 2).mean())), residuals=res, depths=z)
    return cam_r, cam_t, pts, intr, info


def _block_diag(B):
    M = len(B)
    rows = (3 * np.arange(M)[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2)
    cols = (3 * np.arange(M)[:, None, None] + np.arange(3)[None, None, :]).repeat(3, 1)
    return sp.csr_matrix((B.ravel(), (rows.ravel(), cols.ravel())), shape=(3 * M, 3 * M))
