"""Compare CPU vs GPU bundle adjustment on a synthetic scene (accuracy + speed)."""
import sys
import time
import numpy as np


def scene(N, M, seed=0):
    from shapekit import camera as cm
    rng = np.random.default_rng(seed)
    P = rng.uniform(-6, 6, (M, 3)); P[:, 2] = rng.uniform(-1, 1, M)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    C = np.c_[9 * np.cos(ang), 9 * np.sin(ang), np.full(N, 4.0)]
    cr, ct = np.zeros((N, 3)), np.zeros((N, 3))
    for i in range(N):
        z = -C[i] / np.linalg.norm(C[i]); x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x); y = np.cross(z, x)
        R = np.stack([x, y, z]); cr[i] = cm.R_to_rodrigues(R); ct[i] = -R @ C[i]
    intr = np.array([[1500, 960, 540, -0.05, 0.01, 0, 0, 0]], float)
    oc, op = [], []
    for i in range(N):                       # each camera sees ~1/4 of the points, tracks of varying length
        vis = np.where(rng.random(M) < (0.25 if N < 100 else 0.12))[0]
        oc.append(np.full(len(vis), i)); op.append(vis)
    oc, op = np.concatenate(oc), np.concatenate(op)
    uv, z = cm.project(cr[oc], ct[oc], P[op], intr[np.zeros(len(oc), int)])
    ok = (z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
    oc, op, uv = oc[ok], op[ok], uv[ok] + rng.normal(0, 0.5, (ok.sum(), 2))
    return cr, ct, P, intr, oc, op, uv, rng


def main():
    from shapekit.ba import bundle_adjust
    from shapekit.learned.ba_gpu import bundle_adjust_gpu, available
    import torch
    print("CUDA:", available(), torch.cuda.get_device_name(0) if available() else "")
    for N, M in [(60, 8000), (300, 30000)]:
        cr, ct, P, intr, oc, op, uv, rng = scene(N, M)
        cr0, ct0, P0 = cr + rng.normal(0, 0.003, cr.shape), ct + rng.normal(0, 0.02, ct.shape), P + rng.normal(0, 0.02, P.shape)
        i0 = intr.copy(); i0[0, 0] = 1450; i0[0, 3] = 0
        fixed = np.zeros(N, bool); fixed[0] = True
        mask = np.array([1, 1, 1, 1, 1, 0, 0, 0], bool)
        print("\nscene: %d cameras, %d points, %d observations" % (N, M, len(oc)))
        res = {}
        for name, fn in [("GPU", bundle_adjust_gpu), ("CPU", bundle_adjust)]:
            if name == "CPU" and N > 100:
                print("  CPU skipped for the large scene (too slow); GPU result checked against ground truth")
                continue
            t = time.time()
            r, tt, X, I, info = fn(cr0, ct0, P0, i0, np.zeros(N, int), oc, op, uv, fixed_cams=fixed, intr_mask=mask,
                                   iters=15, huber=2.0)
            dt = time.time() - t
            res[name] = (I[0], info["final_rms"])
            print("  %s: %.2fs, %d iters, rms %.3f -> %.3f px, f=%.2f (true 1500), k1=%.4f (true -0.05)" %
                  (name, dt, info["iterations"], info["initial_rms"], info["final_rms"], I[0, 0], I[0, 3]))
        if "CPU" in res:
            d = abs(res["CPU"][1] - res["GPU"][1])
            print("  CPU vs GPU rms difference: %.4f px -> %s" % (d, "MATCH" if d < 0.02 else "MISMATCH"))
        assert abs(res["GPU"][0][0] - 1500) < 5 and res["GPU"][1] < 0.8, "GPU BA did not converge"
    # weighted observations + GCP priors (no fixed camera): GPU must match CPU
    cr, ct, P, intr, oc, op, uv, rng = scene(40, 4000, seed=1)
    cr0, ct0, P0 = cr + rng.normal(0, 0.003, cr.shape), ct + rng.normal(0, 0.02, ct.shape), P + rng.normal(0, 0.02, P.shape)
    pid = rng.choice(len(P), 6, replace=False)
    ow = np.ones(len(oc)); ow[np.isin(op, pid)] = 4.0
    kw = dict(fixed_cams=np.zeros(40, bool), iters=20, huber=2.0, obs_w=ow, prior_idx=pid, prior_X=P[pid],
              prior_info=np.full(6, 1 / 0.005 ** 2))
    print("\nGCP priors + marker weights (40 cameras, 6 GCPs, no fixed camera):")
    out = {}
    for name, fn in [("GPU", bundle_adjust_gpu), ("CPU", bundle_adjust)]:
        r, tt, X, I, info = fn(cr0, ct0, P0, intr, np.zeros(40, int), oc, op, uv, **kw)
        out[name] = X
        print("  %s: rms %.3f px, GCP max residual %.4f, all-point median error %.4f" %
              (name, info["final_rms"], np.abs(X[pid] - P[pid]).max(), np.median(np.linalg.norm(X - P, axis=1))))
    d = np.abs(out["GPU"] - out["CPU"]).max()
    print("  max |GPU - CPU| point difference %.2e -> %s" % (d, "MATCH" if d < 1e-3 else "MISMATCH"))
    assert d < 1e-3
    print("\nBA SELFTEST OK")


if __name__ == "__main__":
    sys.exit(main())
