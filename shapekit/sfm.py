"""Align Photos: incremental Structure-from-Motion with self-calibration.

Pipeline: features -> pair preselection -> matching + verification -> tracks ->
initial pair (essential matrix) -> PnP registration loop -> triangulation -> bundle adjustment
-> outlier filtering (gradual selection by reprojection error) -> final optimisation.
"""
import os
import time
import numpy as np
import cv2

from . import camera as cm
from . import features as F
from .ba import bundle_adjust
from .util import log

DEG = np.pi / 180


class Reconstruction:
    def __init__(self, chunk):
        self.chunk = chunk
        n = len(chunk.cameras)
        self.n = n
        self.gcp = None
        self.marker_trk = {}
        self.allowed = np.array([c.enabled for c in chunk.cameras])
        self.cam_sensor = np.array([c.sensor for c in chunk.cameras])
        self.intr = cm.stack_calib(chunk.sensors)
        self.rvec = np.zeros((n, 3)); self.tvec = np.zeros((n, 3))
        self.reg = np.zeros(n, bool)
        self.fails = np.zeros(n, int)
        try:
            from .learned.ba_gpu import available
            self.use_gpu = available()
        except Exception:
            self.use_gpu = False

    # ------------------------------------------------------------------ data
    def load(self, matches):
        ch = self.chunk
        self.matches = matches
        img, kp, track = F.build_tracks(ch, matches)
        uv = np.zeros((len(img), 2)); rgb = np.zeros((len(img), 3), np.uint8)
        for i in np.unique(img):
            d = np.load(F._feat_path(ch, i))
            m = img == i
            uv[m] = d["xy"][kp[m]]; rgb[m] = d["rgb"][kp[m]]
        from . import masks as MS
        mcams = [i for i in MS.masked_cameras(ch) if i < len(ch.cameras)]
        if mcams:                                  # masks apply to tie points: drop observations inside them
            keep = np.ones(len(img), bool)
            for i in mcams:
                m = img == i
                if m.any():
                    keep[m] = MS.keep_points(ch, i, uv[m])
            img, kp, track, uv, rgb = img[keep], kp[keep], track[keep], uv[keep], rgb[keep]
            log("  masks: %d observations in %d masked photos ignored" % ((~keep).sum(), len(mcams)))
        self.o_img, self.o_kp, self.o_trk, self.o_uv, self.o_rgb = img, kp, track, uv, rgb
        self.T = track.max() + 1
        self.n_feat_tracks = self.T
        self.add_marker_tracks()
        self.o_ok = np.ones(len(self.o_img), bool)
        self.X = np.zeros((self.T, 3)); self.Xv = np.zeros(self.T, bool)
        self.update_normalized()
        log("  %d tracks, %d observations%s" % (self.T, len(self.o_img),
            ", %d marker tracks" % len(self.marker_trk) if self.marker_trk else ""))

    MARKER_WEIGHT = 4.0        # marker projection accuracy 0.5 px vs 1 px for tie points

    def add_marker_tracks(self):
        """Markers (coded targets / manual) become extra tracks with weighted observations. A target seen in
        two different videos ties them together even when feature matching between them failed."""
        self.o_w = np.ones(len(self.o_img))
        self.o_mk = np.zeros(len(self.o_img), bool)
        self.marker_trk = {}
        add = []
        for m in getattr(self.chunk, "markers", None) or []:
            if not m.get("enabled", True):
                continue
            pr = [(int(k), v["xy"]) for k, v in m["projections"].items()   # outliers (misreads) stay out
                  if not v.get("disabled") and not v.get("outlier") and self.chunk.cameras[int(k)].enabled]
            if len(pr) < 2:
                continue
            tid = self.T + len(self.marker_trk)
            self.marker_trk[m["label"]] = tid
            for k, xy in pr:
                add.append((k, tid, xy[0], xy[1]))
        if not add:
            return
        a = np.array(add, float)
        self.o_img = np.concatenate([self.o_img, a[:, 0].astype(int)])
        self.o_kp = np.concatenate([self.o_kp, -np.ones(len(a), int)])
        self.o_trk = np.concatenate([self.o_trk, a[:, 1].astype(int)])
        self.o_uv = np.concatenate([self.o_uv, a[:, 2:4]])
        self.o_rgb = np.concatenate([self.o_rgb, np.tile(np.array([[60, 255, 90]], np.uint8), (len(a), 1))])
        self.o_w = np.concatenate([self.o_w, np.full(len(a), self.MARKER_WEIGHT)])
        self.o_mk = np.concatenate([self.o_mk, np.ones(len(a), bool)])
        self.T += len(self.marker_trk)

    def set_gcp_priors(self):
        """Reference markers -> point priors in the internal frame (needs chunk.transform from Update)."""
        from .markers import get_transform
        self.gcp = None
        ch = self.chunk
        if getattr(ch, "transform", None) is None:
            return 0
        T = get_transform(ch)
        s = float(np.cbrt(abs(np.linalg.det(T[:3, :3])))); R = T[:3, :3] / s
        idx, X0, info = [], [], []
        for m in ch.markers or []:
            ref = m.get("reference")
            if not ref or not m.get("enabled", True) or m["label"] not in self.marker_trk:
                continue
            idx.append(self.marker_trk[m["label"]])
            X0.append(R.T @ (np.asarray(ref["xyz"], float) - T[:3, 3]) / s)
            info.append(1.0 / max(ref.get("accuracy", 0.005) / s, 1e-12) ** 2)
        if idx:
            self.gcp = (np.array(idx), np.array(X0), np.array(info))
        return len(idx)

    def update_normalized(self):
        self.o_xn = np.zeros_like(self.o_uv)
        sen = self.cam_sensor[self.o_img]
        for s in np.unique(sen):
            m = sen == s
            self.o_xn[m] = cm.undistort_points(self.o_uv[m], self.intr[s])

    def px_scale(self, cam):
        s = self.chunk.sensors[self.cam_sensor[cam]]
        return max(s.width, s.height) / 1600.0

    def R(self, i):
        return cm.rodrigues_to_R(self.rvec[i])[0]

    # ------------------------------------------------------------------ init
    def initialize(self, max_pairs=40):
        items = sorted(self.matches.items(), key=lambda kv: -len(kv[1]))
        feats = {}
        best = None
        items = [kv for kv in items if self.allowed[kv[0][0]] and self.allowed[kv[0][1]]]
        for (i, j), m in items[:max_pairs * 3]:
            if len(m) < 100:
                continue
            for c in (i, j):
                if c not in feats:
                    xy = np.load(F._feat_path(self.chunk, c))["xy"]
                    feats[c] = cm.undistort_points(xy, self.intr[self.cam_sensor[c]])
            x1, x2 = feats[i][m[:, 0]], feats[j][m[:, 1]]
            f = self.intr[self.cam_sensor[i], 0]
            E, mask = cv2.findEssentialMat(x1, x2, np.eye(3), cv2.RANSAC, 0.999, 1.5 / f)
            if E is None or E.shape != (3, 3):
                continue
            n, R, t, mask2 = cv2.recoverPose(E, x1, x2, np.eye(3), mask=mask.copy())
            inl = mask2.ravel() > 0
            if inl.sum() < 60:
                continue
            P1 = np.eye(3, 4); P2 = np.hstack([R, t])
            Xh = cv2.triangulatePoints(P1, P2, x1[inl].T, x2[inl].T)
            X = (Xh[:3] / Xh[3]).T
            c2 = -R.T @ t.ravel()
            a = X / np.linalg.norm(X, axis=1, keepdims=True)
            b = (X - c2) / np.linalg.norm(X - c2, axis=1, keepdims=True)
            ang = np.arccos(np.clip((a * b).sum(1), -1, 1))
            med = np.median(ang) / DEG
            Hm, hmask = cv2.findHomography(x1, x2, cv2.RANSAC, 2.0 / f)
            hratio = hmask.sum() / max(mask.sum(), 1) if hmask is not None else 1
            if med < 1.5:
                continue
            score = inl.sum() * min(1.0, med / 5.0) * (1.2 - min(hratio, 1.0))
            if best is None or score > best[0]:
                best = (score, i, j, R, t.ravel(), med, inl.sum())
        if best is None:
            raise RuntimeError("could not find an initial image pair with enough parallax")
        _, i, j, R, t, med, ninl = best
        log("  initial pair %s / %s: %d inliers, median triangulation angle %.1f deg" %
            (self.chunk.cameras[i].label, self.chunk.cameras[j].label, ninl, med))
        self.rvec[i] = 0; self.tvec[i] = 0
        self.rvec[j] = cm.R_to_rodrigues(R); self.tvec[j] = t
        self.reg[i] = self.reg[j] = True
        self.anchor = i
        self.triangulate()
        self.optimize(iters=15, intrinsics=False)
        self.filter()

    # ------------------------------------------------------------------ triangulation
    def triangulate(self, min_angle=1.5, max_err_px=4.0, retri=False):
        reg_obs = self.reg[self.o_img] & self.o_ok
        cand = ~self.Xv if not retri else np.ones(self.T, bool)
        sel = reg_obs & cand[self.o_trk]
        cnt = np.bincount(self.o_trk[sel], minlength=self.T)
        tracks = np.where((cnt >= 2) & cand)[0]
        if len(tracks) == 0:
            return 0
        sel = sel & np.isin(self.o_trk, tracks)
        oi = np.where(sel)[0]
        order = np.argsort(self.o_trk[oi], kind="stable")
        oi = oi[order]
        trk = self.o_trk[oi]
        # cap observations per track for DLT at 8 (first 8)
        start = np.searchsorted(trk, tracks)
        pos = np.arange(len(oi)) - np.repeat(start, cnt[tracks])
        keep = pos < 8
        oi, trk, pos = oi[keep], trk[keep], pos[keep]
        n_per = np.minimum(cnt[tracks], 8)
        Rs = cm.rodrigues_to_R(self.rvec)
        Pm = np.concatenate([Rs, self.tvec[:, :, None]], axis=2)  # (n,3,4)
        newX = np.zeros((len(tracks), 3))
        tidx = np.searchsorted(tracks, trk)
        for k in np.unique(n_per):
            tsel = np.where(n_per == k)[0]
            A = np.zeros((len(tsel), 2 * k, 4))
            m = np.isin(tidx, tsel)
            o = oi[m]; ti = np.searchsorted(tsel, tidx[m]); ps = pos[m]
            P = Pm[self.o_img[o]]
            x = self.o_xn[o]
            A[ti, 2 * ps] = x[:, 0:1] * P[:, 2] - P[:, 0]
            A[ti, 2 * ps + 1] = x[:, 1:2] * P[:, 2] - P[:, 1]
            _, _, Vt = np.linalg.svd(A)
            Xh = Vt[:, -1]
            newX[tsel] = Xh[:, :3] / np.where(np.abs(Xh[:, 3:]) < 1e-12, 1e-12, Xh[:, 3:])
        # validate: depth, reprojection, angle using all registered obs of these tracks
        oall = np.where(reg_obs & np.isin(self.o_trk, tracks))[0]
        ti = np.searchsorted(tracks, self.o_trk[oall])
        uv, z = cm.project(self.rvec[self.o_img[oall]], self.tvec[self.o_img[oall]], newX[ti],
                           self.intr[self.cam_sensor[self.o_img[oall]]])
        err = np.linalg.norm(uv - self.o_uv[oall], axis=1) / np.array([self.px_scale(0)])
        good_obs = (z > 0) & (err < max_err_px)
        ng = np.bincount(ti, good_obs, minlength=len(tracks))
        # max angle between camera rays
        C = -np.einsum("nji,nj->ni", Rs, self.tvec)
        rays = newX[ti] - C[self.o_img[oall]]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
        # angle vs mean ray
        mean = np.zeros((len(tracks), 3))
        np.add.at(mean, ti, rays * good_obs[:, None])
        mean /= np.linalg.norm(mean, axis=1, keepdims=True) + 1e-12
        ang = np.arccos(np.clip((rays * mean[ti]).sum(1), -1, 1))
        amax = np.zeros(len(tracks)); np.maximum.at(amax, ti, np.where(good_obs, ang, 0))
        ok = (ng >= 2) & (2 * amax / DEG >= min_angle) & np.isfinite(newX).all(1)
        self.X[tracks[ok]] = newX[ok]
        self.Xv[tracks[ok]] = True
        bad = ~good_obs & ok[ti]
        self.o_ok[oall[bad]] = False
        return int(ok.sum())

    # ------------------------------------------------------------------ registration
    def register_next(self, batch=None):
        enabled = np.array([c.enabled for c in self.chunk.cameras])
        vis = self.o_ok & self.Xv[self.o_trk] & ~self.reg[self.o_img]
        cnt = np.bincount(self.o_img[vis], minlength=self.n)
        cand = np.where((cnt >= 30) & enabled & self.allowed & ~self.reg & (self.fails < 3))[0]
        if len(cand) == 0:
            return 0
        cand = cand[np.argsort(-cnt[cand])]
        nb = batch or max(1, min(40, int(0.25 * self.reg.sum())))

        def solve(c):
            m = vis & (self.o_img == c)
            o = np.where(m)[0]
            X = self.X[self.o_trk[o]]
            s = self.cam_sensor[c]
            th = 4.0 * self.px_scale(c)
            if cm.model_code(self.intr[s]):
                # fisheye: PnP on undistorted normalised rays (OpenCV's pinhole model can't describe the lens)
                K, D, obs, thr = np.eye(3), None, self.o_xn[o], th / self.intr[s, 0]
            else:
                K, D, obs, thr = cm.K_matrix(self.intr[s]), cm.dist_coeffs(self.intr[s]), self.o_uv[o], th
            ok, rv, tv, inl = cv2.solvePnPRansac(X, obs, K, D, iterationsCount=2000,
                                                 reprojectionError=thr, confidence=0.9999, flags=cv2.SOLVEPNP_SQPNP)
            if not ok or inl is None or len(inl) < max(25, 0.2 * len(o)):
                return c, None
            inl = inl.ravel()
            rv, tv = cv2.solvePnPRefineLM(X[inl], obs[inl], K, D, rv, tv)
            return c, (o, X, s, th, rv.ravel(), tv.ravel())

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 2))) as ex:   # OpenCV releases the GIL
            results = list(ex.map(solve, cand[:nb]))
        added = 0
        for c, res in results:
            if res is None:
                self.fails[c] += 1
                continue
            o, X, s, th, rv, tv = res
            self.rvec[c], self.tvec[c] = rv, tv
            uv, z = cm.project(np.repeat(self.rvec[c:c + 1], len(o), 0), np.repeat(self.tvec[c:c + 1], len(o), 0),
                               X, self.intr[s])
            err = np.linalg.norm(uv - self.o_uv[o], axis=1)
            self.o_ok[o[(err > th) | (z <= 0)]] = False
            self.reg[c] = True
            added += 1
        self.last_added = [c for c, res in results if res is not None]
        return added

    # ------------------------------------------------------------------ optimisation
    def ba_arrays(self):
        m = self.o_ok & self.Xv[self.o_trk] & self.reg[self.o_img]
        cnt = np.bincount(self.o_trk[m], minlength=self.T)
        m &= cnt[self.o_trk] >= 2
        o = np.where(m)[0]
        pts_ids, op = np.unique(self.o_trk[o], return_inverse=True)
        return o, pts_ids, op

    def optimize(self, iters=10, intrinsics=True, fit=("f", "cx", "cy", "k1", "k2", "k3", "p1", "p2"), huber=None,
                 local=None, progress=None):
        """Bundle adjustment. local = list of camera indices to optimise (others fixed, only points they see).
        Runs on the GPU when CUDA is available."""
        o, pts_ids, op = self.ba_arrays()
        if local is not None:
            loc = np.zeros(self.n, bool); loc[list(local)] = True
            pmask = np.zeros(self.T, bool); pmask[self.o_trk[o][loc[self.o_img[o]]]] = True
            o = o[pmask[self.o_trk[o]]]
            pts_ids, op = np.unique(self.o_trk[o], return_inverse=True)
        regs = np.unique(self.o_img[o]) if local is not None else np.where(self.reg)[0]
        cmap = -np.ones(self.n, int); cmap[regs] = np.arange(len(regs))
        fixed = np.zeros(len(regs), bool)
        pri = None
        if self.gcp is not None:
            pmap = -np.ones(self.T, int); pmap[pts_ids] = np.arange(len(pts_ids))
            sel = pmap[self.gcp[0]] >= 0
            if sel.any():
                pri = (pmap[self.gcp[0][sel]], self.gcp[1][sel], self.gcp[2][sel])
        if self.anchor in regs and (pri is None or len(pri[0]) < 3):
            fixed[cmap[self.anchor]] = True          # gauge from the anchor camera unless >= 3 GCPs define it
        if local is not None:
            fixed |= ~np.isin(regs, list(local))
            intrinsics = False
        mask = np.array([n in fit for n in cm.INTR_NAMES]) if intrinsics else np.zeros(8, bool)
        for s in range(len(self.intr)):
            if self.chunk.sensors[s].fixed:
                mask[:] = False
        hub = huber or 2.0 * self.px_scale(0)
        args = (self.rvec[regs], self.tvec[regs], self.X[pts_ids], self.intr, self.cam_sensor[regs],
                cmap[self.o_img[o]], op, self.o_uv[o])
        kw = dict(fixed_cams=fixed, intr_mask=mask, iters=iters, huber=hub, obs_w=self.o_w[o], progress=progress)
        if progress:
            log("  %s: %d cameras, %d points, %d observations, up to %d iterations on %s..." %
                (progress, len(regs), len(pts_ids), len(o), iters, "GPU" if self.use_gpu else "CPU"))
        if pri is not None:
            kw.update(prior_idx=pri[0], prior_X=pri[1], prior_info=pri[2])
        if self.use_gpu:
            from .learned.ba_gpu import bundle_adjust_gpu
            r, t, X, I, info = bundle_adjust_gpu(*args, **kw)
        else:
            r, t, X, I, info = bundle_adjust(*args, **kw)
        self.rvec[regs], self.tvec[regs], self.X[pts_ids] = r, t, X
        if intrinsics and mask.any():
            self.intr = I
            self.update_normalized()
        self.last_info = info
        return info

    def filter(self, max_err_px=3.0, min_angle=1.0):
        o, pts_ids, op = self.ba_arrays()
        uv, z = cm.project(self.rvec[self.o_img[o]], self.tvec[self.o_img[o]], self.X[self.o_trk[o]],
                           self.intr[self.cam_sensor[self.o_img[o]]])
        err = np.linalg.norm(uv - self.o_uv[o], axis=1)
        bad = (err > max_err_px * self.px_scale(0)) | (z <= 0)
        self.o_ok[o[bad]] = False
        o, pts_ids, op = self.ba_arrays()
        keep = np.zeros(self.T, bool); keep[pts_ids] = True
        self.Xv &= keep
        return int(bad.sum())

    # ------------------------------------------------------------------ driver
    # ------------------------------------------------------------------ checkpoint
    _CK_KEYS = ("rvec", "tvec", "reg", "fails", "intr", "X", "Xv", "o_ok")

    def _ck_path(self):
        return os.path.join(self.chunk.dir, "align_checkpoint.npz")

    def _matches_sig(self):
        p = os.path.join(self.chunk.dir, "matches.npz")
        st = os.stat(p)
        return np.array([st.st_size, int(st.st_mtime)], np.int64)

    def save_checkpoint(self, last_global):
        tmp = self._ck_path() + ".tmp.npz"
        np.savez(tmp, sig=self._matches_sig(), anchor=self.anchor, last_global=last_global, n_obs=len(self.o_img),
                 **{k: getattr(self, k) for k in self._CK_KEYS})
        os.replace(tmp, self._ck_path())

    def load_checkpoint(self):
        p = self._ck_path()
        if not os.path.exists(p):
            return None
        d = np.load(p)
        if not np.array_equal(d["sig"], self._matches_sig()) or int(d["n_obs"]) != len(self.o_img) or len(d["reg"]) != self.n:
            return None
        for k in self._CK_KEYS:
            setattr(self, k, d[k].copy())
        self.anchor = int(d["anchor"])
        self.update_normalized()
        return int(d["last_global"])

    def clear_checkpoint(self):
        if os.path.exists(self._ck_path()):
            os.remove(self._ck_path())

    # ------------------------------------------------------------------ driver
    def covisible(self, cams, k=20):
        """Registered cameras sharing the most tie points with `cams`."""
        m = self.o_ok & self.Xv[self.o_trk] & self.reg[self.o_img]
        tr = np.unique(self.o_trk[m & np.isin(self.o_img, cams)])
        sel = m & np.isin(self.o_trk, tr) & ~np.isin(self.o_img, cams)
        cnt = np.bincount(self.o_img[sel], minlength=self.n)
        return [int(i) for i in np.argsort(-cnt)[:k] if cnt[i] > 0]

    def _grow(self, last_global, fit_intrinsics_after=6, checkpoint=True, label=""):
        import time as _t
        t0 = _t.time()
        while True:
            added = self.register_next()
            if added == 0:
                # retry previously failed cameras once structure grew
                if (self.fails > 0).any() and self.triangulate() > 0:
                    self.fails[:] = np.minimum(self.fails, 2)
                    added = self.register_next()
                if added == 0:
                    break
            self.triangulate()
            nreg = int(self.reg.sum())
            full = nreg < 40 or nreg >= 1.2 * last_global
            fit = ("f", "cx", "cy", "k1", "k2") if nreg < 20 else ("f", "cx", "cy", "k1", "k2", "k3", "p1", "p2")
            if full:
                self.optimize(iters=8, intrinsics=nreg >= fit_intrinsics_after, fit=fit)
                last_global = nreg
                kind = "global"
            else:
                new = self.last_added
                self.optimize(iters=6, local=list(new) + self.covisible(new, 25))
                kind = "local"
            self.filter()
            if kind == "global":
                self.triangulate()
            if checkpoint:
                self.save_checkpoint(last_global)
            log("  %sregistered %d/%d cameras, %d tie points, rms %.2f px (%s BA, %.0fs)" %
                (label, nreg, self.n, int(self.Xv.sum()), self.last_info["final_rms"], kind, _t.time() - t0))
        return last_global

    def run(self, fit_intrinsics_after=6, resume=True, components=True):
        last_global = self.load_checkpoint() if resume else None
        if last_global is not None:
            log("  resumed alignment checkpoint: %d cameras already registered" % self.reg.sum())
        else:
            self.initialize()
            last_global = 2
        log("  bundle adjustment on %s" % ("GPU" if self.use_gpu else "CPU"))
        self._grow(last_global, fit_intrinsics_after)
        log("  all cameras that can be registered are in (%d/%d)" % (self.reg.sum(), self.n))
        if components:
            log("  checking for photos that form a separate group (other video / area)...")
            self.align_components()
        log("  re-triangulating tie points with the final camera set...")
        self.triangulate()
        import time as _t
        for k in range(2):
            t = _t.time()
            self.optimize(iters=25, intrinsics=True, progress="final bundle adjustment %d/3" % (k + 1))
            n0 = int(self.Xv.sum())
            self.filter(max_err_px=2.0)
            log("  final bundle adjustment %d/3 done in %.0fs: rms %.2f px; removed outlier projections > 2 px (%d -> %d points)"
                % (k + 1, _t.time() - t, self.last_info["final_rms"], n0, int(self.Xv.sum())))
        t = _t.time()
        info = self.optimize(iters=15, intrinsics=True, progress="final bundle adjustment 3/3")
        log("  final bundle adjustment 3/3 done in %.0fs: rms %.2f px" % (_t.time() - t, info["final_rms"]))
        return info

    # ------------------------------------------------------------------ components / merge by markers
    _STATE = ("rvec", "tvec", "reg", "X", "Xv", "o_ok", "fails")

    def _snap(self):
        d = {k: getattr(self, k).copy() for k in self._STATE}
        d["anchor"] = self.anchor
        return d

    def _restore(self, d):
        for k in self._STATE:
            setattr(self, k, d[k].copy())
        self.anchor = d["anchor"]

    def align_components(self, min_size=6):
        """Cameras left unregistered (e.g. a video whose features never matched the others) are aligned as a
        separate model, then merged into the main one through shared markers / tie points (similarity
        RANSAC on >= 3 common 3D points). Unmergeable components are reported and left unaligned."""
        tried = np.zeros(self.n, bool)
        while True:
            obs_cnt = np.bincount(self.o_img, minlength=self.n)
            rem = self.allowed & ~self.reg & ~tried & (obs_cnt >= 100)
            if rem.sum() < min_size:
                return
            A = self._snap()
            allowed0 = self.allowed.copy()
            self.allowed = rem.copy()
            self.reg[:] = False; self.Xv[:] = False; self.o_ok[:] = True; self.fails[:] = 0
            try:
                self.initialize()
                self._grow(2, checkpoint=False, label="[component] ")
                self.optimize(iters=10, intrinsics=False)
            except RuntimeError as e:
                log("  component of %d cameras could not be initialised (%s)" % (rem.sum(), e))
                self.allowed = allowed0; self._restore(A); tried |= rem
                continue
            B = self._snap()
            self.allowed = allowed0
            tried |= B["reg"] | rem
            if B["reg"].sum() < min_size or not self._merge(A, B):
                self._restore(A)
                if B["reg"].sum() >= min_size:
                    log("  WARNING: a separate component of %d cameras could not be joined: place >= 3 coded "
                        "targets / markers visible in both parts (or add overlap between the videos)" % B["reg"].sum())
                continue

    def _merge(self, A, B):
        from .markers import umeyama
        common = np.where(A["Xv"] & B["Xv"])[0]
        is_mk = common >= self.n_feat_tracks
        log("  merging component (%d cameras) into main model (%d cameras): %d common points (%d markers)" %
            (B["reg"].sum(), A["reg"].sum(), len(common), is_mk.sum()))
        if len(common) < 3:
            return False
        XA, XB = A["X"][common], B["X"][common]
        ext = np.median(np.linalg.norm(XA - np.median(XA, 0), axis=1)) + 1e-9
        thr = 0.03 * ext
        rng = np.random.default_rng(0)
        best = None
        w = np.where(is_mk, 5.0, 1.0)                  # prefer hypotheses supported by markers
        for _ in range(500 if len(common) > 3 else 1):
            smp = rng.choice(len(common), 3, replace=False)
            if np.linalg.matrix_rank(XB[smp] - XB[smp].mean(0), tol=1e-9 * ext) < 2:
                continue
            s, R, t = umeyama(XB[smp], XA[smp])
            err = np.linalg.norm(XB @ (s * R).T + t - XA, axis=1)
            sc = (w * (err < thr)).sum()
            if best is None or sc > best[0]:
                best = (sc, err < thr)
        if best is None or best[1].sum() < 3:
            return False
        inl = best[1]
        s, R, t = umeyama(XB[inl], XA[inl])
        err = np.linalg.norm(XB[inl] @ (s * R).T + t - XA[inl], axis=1)
        log("  similarity B->A: scale %.4f, %d inliers, rms %.4f (%.2f%% of model extent)" %
            (s, inl.sum(), np.sqrt((err ** 2).mean()), 100 * np.sqrt((err ** 2).mean()) / ext))
        # transform component B into A's frame and combine
        cams_B = np.where(B["reg"])[0]
        Rc = cm.rodrigues_to_R(B["rvec"][cams_B])
        Rn = Rc @ R.T
        self.rvec = A["rvec"].copy(); self.tvec = A["tvec"].copy()
        for k, c in enumerate(cams_B):
            self.rvec[c] = cm.R_to_rodrigues(Rn[k])
            self.tvec[c] = s * B["tvec"][c] - Rn[k] @ t
        self.reg = A["reg"] | B["reg"]
        self.X = A["X"].copy(); self.Xv = A["Xv"].copy()
        onlyB = B["Xv"] & ~A["Xv"]
        self.X[onlyB] = B["X"][onlyB] @ (s * R).T + t
        self.Xv |= onlyB
        inB = B["reg"][self.o_img]
        self.o_ok = np.where(inB, B["o_ok"], A["o_ok"])
        self.fails = np.minimum(A["fails"], B["fails"])
        self.anchor = A["anchor"]
        self.triangulate()
        self.optimize(iters=15, intrinsics=False)
        self.filter()
        self.triangulate()
        log("  merged: %d cameras, rms %.2f px" % (self.reg.sum(), self.last_info["final_rms"]))
        # cameras of the merged model may now see enough structure to register more images
        self._grow(int(self.reg.sum()), checkpoint=False)
        return True

    def save_markers(self):
        """Write BA-optimised marker positions + per-projection errors back; drop inconsistent auto markers."""
        ch = self.chunk
        if not self.marker_trk:
            return
        keep = []
        for m in ch.markers:
            tid = self.marker_trk.get(m["label"])
            if tid is None:
                keep.append(m); continue
            ob = np.where((self.o_trk == tid) & self.reg[self.o_img])[0]
            if not self.Xv[tid] or len(ob) == 0:
                if m.get("auto") and ch.aligned_cameras():
                    continue                                  # misread / never triangulated -> drop
                m["position"] = None; keep.append(m); continue
            X = self.X[tid]
            uv, z = cm.project(self.rvec[self.o_img[ob]], self.tvec[self.o_img[ob]], np.repeat(X[None], len(ob), 0),
                               self.intr[self.cam_sensor[self.o_img[ob]]])
            err = np.linalg.norm(uv - self.o_uv[ob], axis=1)
            for k, e, okk in zip(self.o_img[ob], err, self.o_ok[ob]):
                p = m["projections"].get(str(int(k)))
                if p is not None:
                    p["error_px"] = round(float(e), 3); p["outlier"] = not bool(okk)
            inl = err[self.o_ok[ob]]
            if m.get("auto") and len(inl) < 2:
                continue
            m["position"] = X.tolist()
            m["error_px"] = float(np.sqrt((inl ** 2).mean())) if len(inl) else None
            keep.append(m)
        ch.markers = keep

    def save(self):
        ch = self.chunk
        for i, c in enumerate(ch.cameras):
            if self.reg[i]:
                c.rvec, c.tvec = self.rvec[i].copy(), self.tvec[i].copy()
            else:
                c.rvec = c.tvec = None
        for s, sen in enumerate(ch.sensors):
            sen.intr = self.intr[s][:8].copy()
        self.save_markers()
        o, pts_ids, op = self.ba_arrays()
        o = o[~self.o_mk[o]]                        # markers are stored with the markers, not as tie points
        pts_ids, op = np.unique(self.o_trk[o], return_inverse=True)
        # colour = mean of observation colours
        col = np.zeros((len(pts_ids), 3))
        np.add.at(col, op, self.o_rgb[o].astype(float))
        col /= np.bincount(op)[:, None]
        uv, z = cm.project(self.rvec[self.o_img[o]], self.tvec[self.o_img[o]], self.X[self.o_trk[o]],
                           self.intr[self.cam_sensor[self.o_img[o]]])
        err = np.linalg.norm(uv - self.o_uv[o], axis=1)
        ch.save_sparse(points=self.X[pts_ids], colors=col.astype(np.uint8), obs_cam=self.o_img[o],
                       obs_pt=op, obs_uv=self.o_uv[o].astype(np.float32), obs_err=err.astype(np.float32))
        ch.stats["align"] = dict(cameras=int(len(ch.cameras)), aligned=int(self.reg.sum()),
                                 tie_points=int(len(pts_ids)), projections=int(len(o)),
                                 rms_reprojection_px=float(np.sqrt((err ** 2).mean())),
                                 max_reprojection_px=float(err.max()) if len(err) else 0.0)
        return ch.stats["align"]


def align_photos(chunk, accuracy="high", max_features=10000, sequential_window=10, generic_k=12,
                 reset=False, method="sift", reset_features=False, resume=True, retrieval="auto", cross_k=6,
                 use_markers=True, ba_precision=None):
    """reset: discard the current alignment (cameras) -- features and matches are reused if valid.
    reset_features: also recompute features + matches. resume: continue from an interrupted run.
    ba_precision: None (keep current, default 'double') or 'mixed' (faster GPU bundle adjustment)."""
    t0 = time.time()
    _set_ba_precision(ba_precision)
    log("Align Photos [1/6] detecting %s features (%s accuracy) in %d photos (cached features are reused)"
        % (method, accuracy, len(chunk.cameras)))
    method = F.detect_features(chunk, accuracy, max_features, force=reset_features, method=method)
    log("Align Photos [2/6] choosing which photo pairs to match (sequence + retrieval for loop closures)")
    pairs = F.preselect_pairs(chunk, sequential_window, generic_k, retrieval=retrieval, cross_k=cross_k)
    log("  %d candidate pairs" % len(pairs))
    log("Align Photos [3/6] matching %d pairs + geometric verification (cached matches are reused)" % len(pairs))
    F.match_features(chunk, pairs, force=reset_features)
    if use_markers:
        from . import markers as MK
        if reset:
            for c in chunk.cameras:
                c.rvec = c.tvec = None           # detection below must not use a stale alignment
        if not any(m.get("auto") for m in chunk.markers or []):
            log("Align Photos [4/6] detecting coded targets before alignment (they tie videos together)")
            MK.detect_markers(chunk, prealign=True)
        else:
            log("Align Photos [4/6] coded targets already detected - reused")
    else:
        log("Align Photos [4/6] coded targets: skipped")
        chunk.markers = [m for m in chunk.markers or [] if not m.get("auto")]
    log("Align Photos [5/6] building tracks and registering cameras (incremental SfM with bundle adjustment)")
    rec = Reconstruction(chunk)
    rec.load(F.load_matches(chunk))
    if reset:
        rec.clear_checkpoint()
    rec.run(resume=resume and not reset)
    log("Align Photos [6/6] saving cameras, tie points and markers...")
    st = rec.save()
    rec.clear_checkpoint()
    region_from_sparse(chunk)
    chunk.transform = None                       # new internal frame: the old transform is meaningless
    if use_markers:
        from . import markers as MK
        try:                                     # which detections are the same physical target: by 3D position
            MK.regroup_auto_markers(chunk)
        except Exception as e:
            log("  coded target regrouping skipped: %s" % e)
    if use_markers and chunk.markers:
        from . import markers as MK
        try:
            log("  guided marker placement: looking for each target in the photos where its code was unreadable...")
            MK.refine_marker_projections(chunk)  # guided placement in photos where the code was unreadable
        except Exception as e:
            log("  guided marker placement skipped: %s" % e)
        MK.drop_inconsistent_auto(chunk)
        if any(m.get("auto") for m in chunk.markers):
            final_target_adjustment(chunk, ba_precision=ba_precision)
        if any(m.get("reference") for m in chunk.markers) or chunk.scalebars:
            try:
                MK.update_transform(chunk)
            except Exception as e:
                log("  reference not applied: %s" % e)
        st["markers"] = sum(m.get("position") is not None for m in chunk.markers)
    st["features"] = method
    chunk.add_log("Align Photos", time.time() - t0, **{k: v for k, v in st.items()})
    log("Aligned %d/%d cameras, %d tie points, RMS %.2f px" % (st["aligned"], st["cameras"], st["tie_points"],
                                                               st["rms_reprojection_px"]))
    return st


def _set_ba_precision(p):
    if p:
        from .learned import ba_gpu
        ba_gpu.PRECISION = p
        log("  GPU bundle adjustment precision: %s" % p)


def optimize_cameras(chunk, fit=("f", "cx", "cy", "k1", "k2", "k3", "p1", "p2"), max_error_px=None, gcp=True,
                     rounds=3, ba_precision=None):
    """Optimize Cameras on the saved alignment: gradual selection (max_error_px), markers as weighted
    observations and -- if reference coordinates exist -- GCP priors (fit transform, optimise with priors,
    re-fit, repeat). Corrects bending/doming of long video sequences towards the surveyed control."""
    from . import markers as MK
    _set_ba_precision(ba_precision)
    sp_ = chunk.load_sparse()
    rec = Reconstruction(chunk)
    for i, c in enumerate(chunk.cameras):
        if c.aligned:
            rec.rvec[i], rec.tvec[i], rec.reg[i] = c.rvec, c.tvec, True
    keep = np.ones(len(sp_["obs_cam"]), bool)
    if max_error_px:
        keep = sp_["obs_err"] <= max_error_px
    cnt = np.bincount(sp_["obs_pt"][keep], minlength=len(sp_["points"]))
    keep &= cnt[sp_["obs_pt"]] >= 2
    rec.o_img = sp_["obs_cam"][keep]; rec.o_trk = sp_["obs_pt"][keep]; rec.o_uv = sp_["obs_uv"][keep].astype(float)
    rec.o_kp = -np.ones(len(rec.o_img), int)
    rec.o_rgb = sp_["colors"][rec.o_trk]
    rec.T = len(sp_["points"]); rec.n_feat_tracks = rec.T
    rec.add_marker_tracks()
    rec.o_ok = np.ones(len(rec.o_img), bool)
    X = np.zeros((rec.T, 3)); X[:len(sp_["points"])] = sp_["points"]
    Xv = np.zeros(rec.T, bool); Xv[:len(sp_["points"])] = cnt >= 2
    for m in chunk.markers or []:
        tid = rec.marker_trk.get(m["label"])
        if tid is not None and m.get("position") is not None:
            X[tid] = m["position"]; Xv[tid] = True
    rec.X, rec.Xv = X, Xv
    rec.anchor = np.where(rec.reg)[0][0]
    rec.update_normalized()
    rec.triangulate()                       # markers placed since the last alignment
    n_ref = sum(1 for m in chunk.markers or [] if m.get("reference") and m["label"] in rec.marker_trk)
    use_gcp = gcp and n_ref >= 3
    if use_gcp:
        log("Optimize Cameras with %d GCPs (markers with reference coordinates)" % n_ref)
        for k in range(rounds):
            rec.save_markers()
            MK.update_transform(chunk)          # similarity from current marker positions
            rec.set_gcp_priors()
            info = rec.optimize(iters=20, intrinsics=True, fit=fit)
            rep = MK.update_transform(chunk) if k == rounds - 1 else None
        rec.save_markers()
        rep = MK.update_transform(chunk)
        log("  GCP RMSE after optimisation: %.4f" % (chunk.stats["reference"]["marker_rmse"] or 0))
    else:
        info = rec.optimize(iters=30, intrinsics=True, fit=fit)
    st = rec.save()
    if not use_gcp and (any(m.get("reference") for m in chunk.markers or []) or chunk.scalebars):
        try:
            MK.update_transform(chunk)
        except Exception:
            pass
    chunk.add_log("Optimize Cameras", 0, gcp=bool(use_gcp), **st)
    return st


def final_target_adjustment(chunk, ba_precision=None):
    """Cameras, tie points and the cleaned coded targets adjusted together (targets weigh 4x a tie point),
    so the camera poses agree with every confirmed target observation; then the targets are re-placed in
    all photos from their refined 3D positions and re-checked."""
    from . import markers as MK
    n = sum(1 for m in chunk.markers if m.get("position") is not None)
    log("  final adjustment: cameras + tie points + %d coded targets together..." % n)
    try:
        st = optimize_cameras(chunk, gcp=True, ba_precision=ba_precision)
        MK.refine_marker_projections(chunk)
        MK.drop_inconsistent_auto(chunk)
        errs = [m["error_px"] for m in chunk.markers if m.get("error_px") is not None]
        log("  final adjustment done: tie point RMS %s, targets %d, median target error %.2f px"
            % (("%.2f px" % st["rms_reprojection_px"]) if isinstance(st, dict) and "rms_reprojection_px" in st else "n/a",
               len(chunk.markers), float(np.median(errs)) if errs else float("nan")))
    except Exception as e:
        log("  final adjustment skipped: %s" % e)


def region_from_sparse(chunk, lo=2, hi=98, pad=0.1):
    """Auto bounding region from tie points (robust percentiles), axis-aligned to the principal axes."""
    sp_ = chunk.load_sparse()
    X = sp_["points"]
    c = np.median(X, axis=0)
    U, S, Vt = np.linalg.svd(X - c, full_matrices=False)
    R = Vt  # rows = axes
    if np.linalg.det(R) < 0:
        R[2] *= -1
    # orient "up" (3rd axis) towards the cameras
    cams = np.array([chunk.cameras[i].center for i in chunk.aligned_cameras()])
    if len(cams) and ((cams - c) @ R[2]).mean() < 0:
        R[1] *= -1; R[2] *= -1
    L = (X - c) @ R.T
    a, b = np.percentile(L, lo, axis=0), np.percentile(L, hi, axis=0)
    size = (b - a) * (1 + 2 * pad)
    center = c + ((a + b) / 2) @ R
    chunk.region = dict(center=center.tolist(), size=size.tolist(), R=R.tolist())
    return chunk.region
