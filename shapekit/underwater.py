"""Underwater colour correction for textures.

Two modes (Tools > Underwater Colour..., or `shapekit underwater PROJECT ...`):

  wb       White balance from a reference patch (e.g. a sandbag, a grey card): one set of RGB gains for every
           photo. Right at the patch's distance only; water removes red faster with distance.
  seathru  Distance-aware "remove the water", after Akkaynak & Treibitz, Sea-thru (CVPR 2019):
               I = J * exp(-beta * z) + B_inf * (1 - exp(-b * z))          (per colour channel, linear RGB)
           z = camera-to-surface distance of every pixel, rendered from the mesh.
           - backscatter B(z): fitted to the darkest pixels at each distance (dark objects show only backscatter)
           - attenuation beta: from the SAME surface points seen from different distances in different photos
             (multi-view, cf. SUCRe), so it does not depend on what colour the scene is
           - white point: the reference patch (if picked) or grey-world, then exposure normalisation
           J = (I - B(z)) * exp(beta * z) * gains

Settings + estimated parameters live in <chunk>/underwater.json. Build Texture applies them to the photos it
samples (originals are never changed); rebuild the texture after changing them.
"""
import json
import os
import threading
import time

import cv2
import numpy as np

from . import camera as cm
from . import raster
from .util import log, read_image

MODES = ("off", "wb", "seathru")
MODE_LABELS = {"off": "Off", "wb": "White balance (reference patch)",
               "seathru": "Distance-aware: remove the water (Sea-thru style)"}
DEFAULTS = dict(mode="off", patch=None, ref_rgb=None, strength=1.0, max_gain=10.0, params=None)

# ---------------------------------------------------------------------------------------------- colour space
_S2L = np.where(np.arange(256) / 255.0 <= 0.04045, np.arange(256) / 255.0 / 12.92,
                ((np.arange(256) / 255.0 + 0.055) / 1.055) ** 2.4).astype(np.float32)
_LQ = 4096
_l = np.arange(_LQ) / (_LQ - 1.0)
_L2S = np.clip(np.where(_l <= 0.0031308, _l * 12.92, 1.055 * _l ** (1 / 2.4) - 0.055) * 255 + 0.5, 0, 255).astype(np.uint8)


def to_linear(img_u8):
    return _S2L[img_u8]


def to_srgb(lin):
    q = np.clip(lin * (_LQ - 1) + 0.5, 0, _LQ - 1).astype(np.uint16)
    return _L2S[q]


def srgb_to_linear_rgb(rgb):
    return _S2L[np.clip(np.round(np.asarray(rgb, float)), 0, 255).astype(int)].astype(float)


# ---------------------------------------------------------------------------------------------- settings
def settings_path(chunk):
    return os.path.join(chunk.dir, "underwater.json")


def load_settings(chunk):
    d = dict(DEFAULTS)
    try:
        with open(settings_path(chunk), encoding="utf-8") as f:
            d.update(json.load(f))
    except (OSError, ValueError):
        pass
    if d.get("mode") not in MODES:
        d["mode"] = "off"
    return d


def save_settings(chunk, d):
    p = settings_path(chunk)
    out = {k: d.get(k, v) for k, v in DEFAULTS.items()}
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, p)


def _settings_key(s):
    """What the estimated parameters depend on (re-estimate when this changes)."""
    return json.dumps([s.get("mode"), s.get("patch"), s.get("ref_rgb"), float(s.get("strength", 1.0)),
                       float(s.get("max_gain", 10.0))], sort_keys=True)


# ---------------------------------------------------------------------------------------------- geometry
_mesh_cache = {}
_raster_lock = threading.Lock()


def _mesh(chunk):
    from .mesh import load_mesh
    p = os.path.join(chunk.dir, "mesh.ply")
    if not os.path.exists(p):
        return None
    key = (p, os.path.getmtime(p))
    if key not in _mesh_cache:
        _mesh_cache.clear()
        V, F, _ = load_mesh(chunk)
        _mesh_cache[key] = (np.asarray(V, np.float64), np.asarray(F, np.int64))
    return _mesh_cache[key]


def undistorted(chunk, i, scale=1.0, img=None):
    """Undistorted pinhole view of camera i (as Build Texture uses it) and its K."""
    cam = chunk.cameras[i]
    sen = chunk.sensors[cam.sensor]
    img = read_image(chunk.abspath(cam)) if img is None else img
    m1, m2, K = cm.undistort_maps(sen.calib(), sen.width, sen.height, scale)
    if scale < 1:
        img = cv2.GaussianBlur(img, (0, 0), 0.5 / scale * 0.8)          # anti-alias before sampling down
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), K


def range_map(chunk, i, K, w, h, work=0.25):
    """Camera-to-surface distance (scene units) of every pixel of an undistorted (w,h) view with intrinsics K.
    Rendered from the mesh at `work` x resolution; holes take the nearest surface distance.
    Returns (range (h,w), covered mask at the work resolution) or None if the mesh is not in view."""
    mesh = _mesh(chunk)
    if mesh is None:
        raise RuntimeError("the distance-aware correction needs a mesh: run Build Mesh first")
    V, F = mesh
    cam = chunk.cameras[i]
    ww, hh = max(8, int(w * work)), max(8, int(h * work))
    Ks = K.copy().astype(np.float64)
    Ks[0] *= ww / w; Ks[1] *= hh / h
    uv, z = raster.project_vertices(V, cam.rvec, cam.tvec, Ks)
    with _raster_lock:
        out = raster.rasterize(uv, z, F, ww, hh)
    zb = np.asarray(out[1], np.float32)
    ok = np.isfinite(zb) & (zb > 0)
    if not ok.any():
        return None
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
    rx = (xs - Ks[0, 2]) / Ks[0, 0]; ry = (ys - Ks[1, 2]) / Ks[1, 1]
    r = zb * np.sqrt(1 + rx * rx + ry * ry)
    if (~ok).any():
        _, lab = cv2.distanceTransformWithLabels((~ok).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        yy, xx = np.nonzero(ok)
        src = np.zeros(lab.max() + 1, np.int64); src[lab[yy, xx]] = yy * ww + xx
        r = r.reshape(-1)[src[lab].reshape(-1)].reshape(hh, ww)
    r = cv2.GaussianBlur(r.astype(np.float32), (0, 0), 1.0)
    return cv2.resize(r, (w, h), interpolation=cv2.INTER_LINEAR), ok


def photo_rect_to_undistorted(chunk, i, rect):
    """A rectangle in original-photo pixels -> bounding box in the undistorted view (what the patch uses)."""
    cam = chunk.cameras[i]; sen = chunk.sensors[cam.sensor]
    x0, y0, x1, y1 = rect
    xs, ys = np.linspace(x0, x1, 5), np.linspace(y0, y1, 5)
    pts = np.array([(x, y) for x in xs for y in ys], float)
    xn = cm.undistort_points(pts, sen.calib())
    K = cm.K_matrix(sen.intr)
    u = xn[:, 0] * K[0, 0] + K[0, 2]; v = xn[:, 1] * K[1, 1] + K[1, 2]
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]


# ---------------------------------------------------------------------------------------------- correction
def _apply_lin(lin_bgr, z, params, strength=1.0, max_gain=10.0):
    """lin_bgr (h,w,3) float32 linear, z (h,w) range or None -> corrected linear BGR."""
    g = np.asarray(params["gains"], np.float32)[::-1]                    # stored RGB -> BGR
    if params.get("kind") == "wb" or z is None:
        return lin_bgr * g[None, None]
    Binf = np.asarray(params["Binf"], np.float32)[::-1] * strength
    b = np.asarray(params["b"], np.float32)[::-1]
    beta = np.asarray(params["beta"], np.float32)[::-1] * strength
    zz = z[..., None].astype(np.float32)
    B = Binf * (1 - np.exp(-b * zz))
    D = np.maximum(lin_bgr - B, 0)
    return D * np.minimum(np.exp(beta * zz), max_gain) * g[None, None]


class Corrector:
    """Applies the chunk's saved underwater correction to undistorted views (used by Build Texture)."""

    def __init__(self, chunk, settings=None):
        self.chunk = chunk
        self.s = settings or load_settings(chunk)
        self.active = self.s["mode"] != "off"
        p = self.s.get("params")
        if self.active and (not p or p.get("key") != _settings_key(self.s)):
            log("Underwater colour (%s): estimating the correction..." % self.s["mode"])
            p = estimate(chunk, self.s)
            self.s["params"] = p
            save_settings(chunk, self.s)
        self.params = p

    def describe(self):
        if not self.active:
            return "off"
        p = self.params
        if p["kind"] == "wb":
            return "white balance, gains R %.2f G %.2f B %.2f" % tuple(p["gains"])
        return ("remove water: beta R %.3g G %.3g B %.3g /unit, backscatter %.3f %.3f %.3f, strength %.2f"
                % (*p["beta"], *p["Binf"], self.s.get("strength", 1.0)))

    def correct_samples(self, rgb_u8, ranges):
        """Correct individual pixel colours (n,3 RGB uint8) whose surface distances are known (n,)."""
        if not self.active:
            return rgb_u8
        lin = to_linear(np.asarray(rgb_u8, np.uint8)[:, ::-1])[:, None]          # (n,1,3) BGR
        z = None if self.params["kind"] == "wb" else np.asarray(ranges, np.float32)[:, None]
        out = _apply_lin(lin, z, self.params, float(self.s.get("strength", 1.0)), float(self.s.get("max_gain", 10.0)))
        return to_srgb(out)[:, 0, ::-1]

    def apply(self, i, und_bgr_u8, K):
        if not self.active:
            return und_bgr_u8
        h, w = und_bgr_u8.shape[:2]
        z = None
        if self.params["kind"] == "seathru":
            rm = range_map(self.chunk, i, K, w, h)
            z = rm[0] if rm is not None else None
        out = _apply_lin(to_linear(und_bgr_u8), z, self.params, float(self.s.get("strength", 1.0)),
                         float(self.s.get("max_gain", 10.0)))
        return to_srgb(out)


# ---------------------------------------------------------------------------------------------- estimation
def _patch_mean(chunk, settings, corrected_params=None, strength=1.0, max_gain=10.0):
    """Mean linear RGB inside the reference patch (optionally after water removal)."""
    pt = settings.get("patch")
    if not pt:
        return None
    i = int(pt["camera"])
    x0, y0, x1, y1 = [float(v) for v in pt["rect"]]
    und, K = undistorted(chunk, i, 1.0)
    h, w = und.shape[:2]
    x0, x1 = int(max(0, min(x0, x1))), int(min(w, max(x0, x1)))
    y0, y1 = int(max(0, min(y0, y1))), int(min(h, max(y0, y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("the reference patch is empty")
    lin = to_linear(und[y0:y1, x0:x1])
    if corrected_params is not None:
        z = range_map(chunk, i, K, w, h)[0][y0:y1, x0:x1]
        lin = _apply_lin(lin, z, dict(corrected_params, gains=[1, 1, 1]), strength, max_gain)
    m = lin.reshape(-1, 3)
    m = m[(m.max(1) < 0.98)]                                 # ignore clipped pixels
    if len(m) < 4:
        raise ValueError("the reference patch is over-exposed")
    return np.median(m, axis=0)[::-1]                        # RGB


def _white_point(chunk, s, samples_rgb, patch_rgb):
    """RGB gains: patch -> its true colour (neutral by default), else grey-world; then exposure."""
    if patch_rgb is not None:
        lum = float(np.mean(patch_rgb))
        if s.get("ref_rgb"):
            t = srgb_to_linear_rgb(s["ref_rgb"])
            t = t / max(t.mean(), 1e-6) * lum
        else:
            t = np.full(3, lum)
        gains = t / np.maximum(patch_rgb, 1e-6)
        src = "reference patch"
    else:
        m = np.mean(samples_rgb, axis=0)
        gains = m.mean() / np.maximum(m, 1e-6)
        src = "grey-world (no reference patch)"
    v = (samples_rgb * gains[None]).max(1)
    p99 = float(np.percentile(v, 99.5)) if len(v) else 1.0
    gains = gains * (0.92 / max(p99, 1e-6))                  # brightest surfaces just below white
    return gains, src


def _cameras_for_estimate(chunk, n=40):
    al = chunk.aligned_cameras()
    if len(al) > n:
        al = [al[int(k)] for k in np.linspace(0, len(al) - 1, n)]
    return al


def estimate(chunk, settings=None, max_cams=40, progress=True):
    """Estimate the correction for the chunk's settings. Returns the params dict (also kept in settings)."""
    s = settings or load_settings(chunk)
    t0 = time.time()
    mode = s["mode"]
    if mode == "off":
        return None
    cams = _cameras_for_estimate(chunk, max_cams)
    if not cams:
        raise RuntimeError("align the photos first")
    rng = np.random.default_rng(0)
    if mode == "wb":
        smp = []
        for i in cams[:: max(1, len(cams) // 12)]:
            und, _ = undistorted(chunk, i, 0.25)
            lin = to_linear(und).reshape(-1, 3)[:, ::-1]
            smp.append(lin[rng.choice(len(lin), min(len(lin), 20000), replace=False)])
        smp = np.concatenate(smp)
        pm = _patch_mean(chunk, s)
        gains, src = _white_point(chunk, s, smp, pm)
        p = dict(kind="wb", gains=gains.tolist(), white=src, key=_settings_key(s),
                 estimated=time.strftime("%Y-%m-%d %H:%M:%S"))
        log("  white balance from %s: gains R %.2f G %.2f B %.2f" % (src, *gains))
        return p

    # ---- seathru: gather pixels with distances, and multi-view observations of mesh vertices
    mesh = _mesh(chunk)
    if mesh is None:
        raise RuntimeError("the distance-aware correction needs a mesh: run Build Mesh first")
    V, F = mesh
    vs = rng.choice(len(V), min(len(V), 60000), replace=False)
    pix_z, pix_c = [], []
    obs_v, obs_z, obs_c = [], [], []
    for n, i in enumerate(cams):
        und, K = undistorted(chunk, i, 0.25)
        h, w = und.shape[:2]
        rm = range_map(chunk, i, K, w, h, work=1.0)
        if rm is None:
            continue
        z, ok = rm
        lin = to_linear(und)[:, :, ::-1]                      # RGB
        idx = np.flatnonzero(ok.reshape(-1))
        if len(idx) > 25000:
            idx = rng.choice(idx, 25000, replace=False)
        pix_z.append(z.reshape(-1)[idx]); pix_c.append(lin.reshape(-1, 3)[idx])
        # vertices: projected + depth test against the rendered surface
        cam = chunk.cameras[i]
        uv, zc = raster.project_vertices(V[vs], cam.rvec, cam.tvec, K)
        u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
        inside = (zc > 0) & (u >= 1) & (v >= 1) & (u < w - 1) & (v < h - 1)
        if inside.any():
            C = cam.center
            rr = np.linalg.norm(V[vs[inside]] - C, axis=1)
            zr = z[v[inside], u[inside]]
            vis = ok[v[inside], u[inside]] & (np.abs(zr - rr) < 0.02 * rr)
            sel = np.flatnonzero(inside)[vis]
            obs_v.append(vs[sel]); obs_z.append(rr[vis]); obs_c.append(lin[v[sel], u[sel]])
        if progress and (n + 1) % 10 == 0:
            log("  underwater colour: sampled %d/%d photos" % (n + 1, len(cams)))
    if not pix_z:
        raise RuntimeError("no photo sees the mesh")
    pz = np.concatenate(pix_z); pc = np.concatenate(pix_c)

    # ---- backscatter: darkest 1 % (RGB sum) per distance bin, fit Binf (1 - exp(-b z)) per channel
    from scipy.optimize import curve_fit
    edges = np.quantile(pz, np.linspace(0.02, 0.98, 13))
    bz, bc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pz >= a) & (pz < b)
        if m.sum() < 100:
            continue
        c = pc[m]
        dark = c[np.argsort(c.sum(1))[: max(5, int(0.01 * len(c)))]]
        bz.append(float(np.median(pz[m]))); bc.append(np.median(dark, axis=0))
    bz = np.array(bz); bc = np.array(bc)
    zmed = float(np.median(pz))
    Binf, bb = np.zeros(3), np.full(3, 1.0 / max(zmed, 1e-6))
    for ch in range(3):
        f = lambda zz, A, k: A * (1 - np.exp(-k * zz))           # noqa: E731
        try:
            (A, k), _ = curve_fit(f, bz, bc[:, ch], p0=[max(bc[:, ch].max(), 1e-3), 1.0 / max(zmed, 1e-6)],
                                  bounds=([0, 0], [1, 50.0 / max(zmed, 1e-6)]), maxfev=5000)
            Binf[ch], bb[ch] = A, k
        except Exception:
            Binf[ch], bb[ch] = float(np.min(bc[:, ch])) if len(bc) else 0.0, 50.0 / max(zmed, 1e-6)

    def backsc(zz):
        return Binf[None] * (1 - np.exp(-bb[None] * zz[:, None]))

    # ---- attenuation: same vertex at different distances -> within-vertex regression of log(I - B) on z
    beta, how = np.zeros(3), "multi-view"
    if obs_v:
        ov = np.concatenate(obs_v); oz = np.concatenate(obs_z); oc = np.concatenate(obs_c)
        good = (oc.max(1) < 0.98)
        ov, oz, oc = ov[good], oz[good], oc[good]
        D = oc - backsc(oz)
        y = np.log(np.maximum(D, 1e-4))
        good = (D > 0.004).all(1)
        ov, oz, y = ov[good], oz[good], y[good]
        uniq, inv, cnt = np.unique(ov, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= 2
        ov, oz, y, inv = ov[keep], oz[keep], y[keep], inv[keep]
        ok_fit = False
        if len(oz) > 500:
            for it in range(3):
                nb = inv.max() + 1
                zc = oz - (np.bincount(inv, oz, nb) / np.maximum(np.bincount(inv, None, nb), 1))[inv]
                yc = y - (np.stack([np.bincount(inv, y[:, c], nb) for c in range(3)], 1)
                          / np.maximum(np.bincount(inv, None, nb), 1)[:, None])[inv]
                den = float((zc * zc).sum())
                if den < 1e-9:
                    break
                beta = -(zc[:, None] * yc).sum(0) / den
                res = yc + beta[None] * zc[:, None]
                mad = np.median(np.abs(res), axis=0) + 1e-6
                inl = (np.abs(res) < 3.5 * 1.4826 * mad[None]).all(1)
                ok_fit = True
                if inl.all():
                    break
                ov, oz, y, inv = ov[inl], oz[inl], y[inl], np.unique(inv[inl], return_inverse=True)[1]
            spread = float(np.sqrt(den / max(len(oz), 1))) if ok_fit else 0.0
            if not ok_fit or spread < 0.03 * zmed:
                ok_fit = False
        if not ok_fit:
            beta[:] = 0
    if not beta.any():
        # fallback: grey-world per distance bin (assumes the average scene colour does not change with distance)
        how = "per-distance grey-world (too little multi-view range spread)"
        D = np.maximum(pc - backsc(pz), 1e-4)
        zs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (pz >= a) & (pz < b)
            if m.sum() > 200:
                zs.append(np.median(pz[m])); ys.append(np.log(np.median(D[m], axis=0)))
        if len(zs) >= 3:
            A = np.stack([np.ones(len(zs)), zs], 1)
            beta = -np.linalg.lstsq(A, np.array(ys), rcond=None)[0][1]
    beta = np.clip(beta, 0, 20.0 / max(zmed, 1e-6))
    if how == "multi-view":
        Binf, bb, beta, how = _refine_joint(np.concatenate(obs_v), np.concatenate(obs_z), np.concatenate(obs_c),
                                            Binf, bb, beta, zmed, how)
    params = dict(kind="seathru", Binf=Binf.tolist(), b=bb.tolist(), beta=beta.tolist(), gains=[1.0, 1.0, 1.0],
                  median_range=zmed, attenuation_from=how)

    # ---- white point on the water-free colours
    st, mg = float(s.get("strength", 1.0)), float(s.get("max_gain", 10.0))
    Jlin = _apply_lin(pc[:, None, ::-1].astype(np.float32), pz[:, None].astype(np.float32),
                      params, st, mg)[:, 0, ::-1]
    pm = _patch_mean(chunk, s, params, st, mg)
    gains, src = _white_point(chunk, s, Jlin, pm)
    params.update(gains=gains.tolist(), white=src, key=_settings_key(s), estimated=time.strftime("%Y-%m-%d %H:%M:%S"),
                  photos=len(cams), seconds=round(time.time() - t0, 1))
    log("  water: backscatter R %.3f G %.3f B %.3f, attenuation R %.3g G %.3g B %.3g per unit (%s), median distance %.3g"
        % (*Binf, *beta, how, zmed))
    log("  white point from %s: gains R %.2f G %.2f B %.2f (%.0fs)" % (src, *gains, time.time() - t0))
    return params


def _refine_joint(ov, oz, oc, Binf, bb, beta, zmed, how, max_obs=60000):
    """Fit the full model I_ij = J_i exp(-beta z_ij) + Binf (1 - exp(-b z_ij)) to multi-view observations
    (vertex i seen at distance z_ij); the unknown seabed colours J_i are solved in closed form inside the fit
    (variable projection), so only the 3 water parameters per channel are optimised. The dark-pixel backscatter estimate is biased high when no pixel is truly black; this
    uses the range spread of the same points instead."""
    from scipy.optimize import least_squares
    good = oc.max(1) < 0.98
    ov, oz, oc = ov[good], oz[good], oc[good]
    uniq, inv, cnt = np.unique(ov, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2
    ov, oz, oc, inv = ov[keep], oz[keep], oc[keep], np.unique(inv[keep], return_inverse=True)[1]
    if len(oz) < 500:
        return Binf, bb, beta, how
    if len(oz) > max_obs:
        rng = np.random.default_rng(1)
        pts = rng.choice(inv.max() + 1, max(50, int((inv.max() + 1) * max_obs / len(oz))), replace=False)
        m = np.isin(inv, pts)
        oz, oc, inv = oz[m], oc[m], np.unique(inv[m], return_inverse=True)[1]
    nb = inv.max() + 1
    zmax = 50.0 / max(zmed, 1e-6)
    B0, b0, be0 = Binf.copy(), bb.copy(), beta.copy()
    out_B, out_b, out_be = Binf.copy(), bb.copy(), beta.copy()
    improved = 0
    for c in range(3):
        I = oc[:, c].astype(np.float64)

        def resid(x):
            # variable projection: for given water parameters the best seabed colour of every point is closed-form
            e = np.exp(-x[2] * oz)
            D = I - x[0] * (1 - np.exp(-x[1] * oz))
            J = np.maximum(np.bincount(inv, D * e, nb) / np.maximum(np.bincount(inv, e * e, nb), 1e-12), 0)
            return J[inv] * e - D
        best = None
        for x0 in ([B0[c], b0[c], be0[c]], [0.5 * B0[c], b0[c], 0.7 * be0[c]], [B0[c], 0.5 * b0[c], 1.3 * be0[c]]):
            try:
                r = least_squares(resid, np.clip(x0, [0, 1e-6, 0], [1, zmax, 20.0 / max(zmed, 1e-6)]),
                                  bounds=([0, 1e-6, 0], [1, zmax, 20.0 / max(zmed, 1e-6)]),
                                  loss="soft_l1", f_scale=0.01, max_nfev=400)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        cost0 = float(np.mean(resid(np.array([B0[c], b0[c], be0[c]])) ** 2))
        if best is not None and float(np.mean(best.fun ** 2)) < cost0:
            out_B[c], out_b[c], out_be[c] = best.x
            improved += 1
    return out_B, out_b, out_be, (how + " + joint refinement" if improved else how)


def preview(chunk, i, settings=None, scale=0.35):
    """(before, after) undistorted BGR uint8 previews of camera i with the given settings."""
    s = settings or load_settings(chunk)
    und, K = undistorted(chunk, i, scale)
    if s["mode"] == "off":
        return und, und
    c = Corrector(chunk, s)
    return und, c.apply(i, und, K)
