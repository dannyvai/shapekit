"""Benchmark the feature detector / matcher combinations on your own images, to choose the best for a project.

    shapekit bench-features PROJECT_OR_FOLDER [--pairs 5] [--methods sift,superpoint,aliked,disk]
                            [--accuracy high] [--max-features 4096] [--out report.html]
    GUI: Tools > Benchmark Feature Methods...

Runs exactly the code Align Photos uses (RootSIFT + FLANN on CPU; SuperPoint / ALIKED / DISK + LightGlue on the
GPU; MAGSAC++ verification) on a few image pairs spread over the project, at several baselines, and writes a
self-contained HTML report: every pair x method drawn with its verified matches, plus a table and a
recommendation.

Measured per pair: keypoints, matches before / after geometric verification, inlier ratio, coverage (share of a
10 x 10 grid over the image holding verified matches - well-spread matches make a stronger alignment) and time.
If the project is aligned, each match is also checked against the known camera poses: the epipolar error says
how many of a method's "verified" matches are actually right.
"""
import base64
import glob
import html
import os
import time

import cv2
import numpy as np

from .. import camera as cm
from ..util import log, read_image

ALL_METHODS = ("sift", "superpoint", "aliked", "disk")
LABEL = {"sift": "RootSIFT + FLANN", "superpoint": "SuperPoint + LightGlue", "aliked": "ALIKED + LightGlue",
         "disk": "DISK + LightGlue"}
GAPS = (1, 3, 6, 10, 15)           # capture-order distance between the two images of a pair
MIN_INLIERS = 15


# ---------------------------------------------------------------------------------------------- inputs
class _Img:
    def __init__(self, path, label, cam=None, sensor=None):
        self.path, self.label, self.cam, self.sensor = path, label, cam, sensor


def _inputs(src):
    """Images (+ poses when src is a project) in capture order."""
    if os.path.exists(os.path.join(src, "project.json")):
        from ..project import Project
        P = Project.open(src)
        ch = P.chunk(0)
        imgs = [_Img(ch.abspath(c), c.label, c if c.aligned else None, ch.sensors[c.sensor])
                for c in ch.cameras if getattr(c, "enabled", True)]
        imgs.sort(key=lambda m: m.label)
        return imgs, os.path.join(src, "feature_benchmark.html")
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    files = sorted(f for f in glob.glob(os.path.join(src, "*")) if f.lower().endswith(exts))
    return [_Img(f, os.path.splitext(os.path.basename(f))[0]) for f in files], os.path.join(src, "feature_benchmark.html")


def choose_pairs(imgs, n=5):
    """n pairs spread over the sequence, with growing baselines (gaps 1, 3, 6, 10, 15 in capture order)."""
    if len(imgs) < 2:
        raise ValueError("need at least 2 images")
    pairs = []
    anchors = np.linspace(0, len(imgs) - 2, n).round().astype(int)
    for k, a in enumerate(anchors):
        g = GAPS[k % len(GAPS)]
        b = min(len(imgs) - 1, a + g)
        if b == a:
            continue
        pairs.append((int(a), int(b)))
    return list(dict.fromkeys(pairs))


# ---------------------------------------------------------------------------------------------- methods
def _extract(method, img, accuracy, max_features):
    from ..features import ACCURACY, learned_resize
    scale = ACCURACY[accuracy] if isinstance(accuracy, str) else float(accuracy)
    if method == "sift":
        sm = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else img
        gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        kps, desc = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02).detectAndCompute(gray, None)
        if desc is None or not len(kps):
            return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
        xy = (np.array([k.pt for k in kps], np.float32) + 0.5) / scale - 0.5
        desc = desc / (np.abs(desc).sum(1, keepdims=True) + 1e-7)
        return xy, np.sqrt(desc).astype(np.float32)
    from .. import learned
    lf = learned.get(method, max_features, None)
    f = lf.extract(img, resize=learned_resize(lf, img.shape[1], img.shape[0], scale))
    return f["xy"], f["desc"]


def _match(method, f0, f1, size0, size1, max_features):
    if method == "sift":
        from ..features import match_pair
        return match_pair(f0[1], f1[1])
    from .. import learned
    m, _ = learned.get(method, max_features, None).match(f0, f1, size0, size1)
    return m


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ---------------------------------------------------------------------------------------------- metrics
def _coverage(xy, w, h, n=10):
    if not len(xy):
        return 0.0
    gx = np.clip((xy[:, 0] / w * n).astype(int), 0, n - 1)
    gy = np.clip((xy[:, 1] / h * n).astype(int), 0, n - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / float(n * n)


def _epipolar_px(A, B, xy0, xy1):
    """Symmetric point-to-epipolar-line distance (pixels) of matches, from the known camera poses."""
    R0 = cm.rodrigues_to_R(np.asarray(A.cam.rvec, float))[0]; t0 = np.asarray(A.cam.tvec, float)
    R1 = cm.rodrigues_to_R(np.asarray(B.cam.rvec, float))[0]; t1 = np.asarray(B.cam.tvec, float)
    R = R1 @ R0.T
    t = t1 - R @ t0
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    n0 = cm.undistort_points(xy0, A.sensor.calib()); n1 = cm.undistort_points(xy1, B.sensor.calib())
    h0 = np.c_[n0, np.ones(len(n0))]; h1 = np.c_[n1, np.ones(len(n1))]
    l1 = h0 @ E.T                          # lines in image 1
    l0 = h1 @ E                            # lines in image 0
    d1 = np.abs((h1 * l1).sum(1)) / np.maximum(np.hypot(l1[:, 0], l1[:, 1]), 1e-12)
    d0 = np.abs((h0 * l0).sum(1)) / np.maximum(np.hypot(l0[:, 0], l0[:, 1]), 1e-12)
    return 0.5 * (d0 * A.sensor.intr[0] + d1 * B.sensor.intr[0])


# ---------------------------------------------------------------------------------------------- drawing
def _jpg(canvas):
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _draw(img0, img1, xy0, xy1, inl, w_out=1100, max_lines=150):
    """Three views of one pair x method: 'lines' (verified green, rejected red), 'points' (each verified match a
    dot of the same colour in both photos, colour = position in the first photo) and 'coverage' (where in each
    photo the verified matches are, as a heat map over a 10 x 10 grid)."""
    h0, w0 = img0.shape[:2]; h1, w1 = img1.shape[:2]
    s0 = (w_out / 2) / w0; s1 = (w_out / 2) / w1
    a = cv2.resize(img0, (int(w0 * s0), int(h0 * s0)), interpolation=cv2.INTER_AREA)
    b = cv2.resize(img1, (int(w1 * s1), int(h1 * s1)), interpolation=cv2.INTER_AREA)
    H = max(a.shape[0], b.shape[0]); off = a.shape[1]

    def base(dim=0.6):
        c = np.full((H, a.shape[1] + b.shape[1], 3), 30, np.uint8)
        c[:a.shape[0], :off] = (a * dim).astype(np.uint8)
        c[:b.shape[0], off:] = (b * dim).astype(np.uint8)
        return c
    P0 = np.c_[xy0[:, 0] * s0, xy0[:, 1] * s0].astype(int) if len(xy0) else np.zeros((0, 2), int)
    P1 = np.c_[xy1[:, 0] * s1 + off, xy1[:, 1] * s1].astype(int) if len(xy1) else np.zeros((0, 2), int)
    rng = np.random.default_rng(0)
    in_idx, out_idx = np.flatnonzero(inl), np.flatnonzero(~inl)
    # lines: fewer, blended so the photos stay visible
    lines = base(0.7); over = lines.copy()
    for k in rng.permutation(out_idx)[:max_lines // 3]:
        cv2.line(over, tuple(P0[k]), tuple(P1[k]), (70, 70, 230), 1, cv2.LINE_AA)
    for k in rng.permutation(in_idx)[:max_lines]:
        cv2.line(over, tuple(P0[k]), tuple(P1[k]), (90, 230, 120), 1, cv2.LINE_AA)
    lines = cv2.addWeighted(over, 0.75, lines, 0.25, 0)
    # points: same colour = same match
    pts = base(0.55)
    for k in in_idx:
        hue = int(179 * xy0[k, 0] / max(w0, 1)); val = int(150 + 105 * xy0[k, 1] / max(h0, 1))
        col = cv2.cvtColor(np.uint8([[[hue, 230, val]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.circle(pts, tuple(P0[k]), 3, col, -1, cv2.LINE_AA)
        cv2.circle(pts, tuple(P1[k]), 3, col, -1, cv2.LINE_AA)
    # coverage: verified-match density per cell of a 10 x 10 grid, both photos
    cov = base(0.8)
    for P, x0, wpx, hpx in ((P0, 0, a.shape[1], a.shape[0]), (P1, off, b.shape[1], b.shape[0])):
        n = 10
        grid = np.zeros((n, n))
        for k in in_idx:
            gx = min(n - 1, max(0, int((P[k, 0] - x0) / wpx * n))); gy = min(n - 1, max(0, int(P[k, 1] / hpx * n)))
            grid[gy, gx] += 1
        top = grid.max() or 1
        layer = cov.copy()
        for gy in range(n):
            for gx in range(n):
                if grid[gy, gx]:
                    t = grid[gy, gx] / top
                    col = (int(60 + 40 * t), int(140 + 115 * t), int(60 + 40 * t))
                    cv2.rectangle(layer, (x0 + int(gx * wpx / n), int(gy * hpx / n)),
                                  (x0 + int((gx + 1) * wpx / n) - 1, int((gy + 1) * hpx / n) - 1), col, -1)
        cov = cv2.addWeighted(layer, 0.38, cov, 0.62, 0)
        for g in range(1, n):
            cv2.line(cov, (x0 + int(g * wpx / n), 0), (x0 + int(g * wpx / n), hpx), (90, 90, 90), 1)
            cv2.line(cov, (x0, int(g * hpx / n)), (x0 + wpx, int(g * hpx / n)), (90, 90, 90), 1)
    cv2.line(cov, (off, 0), (off, H), (20, 20, 20), 2)
    return dict(lines=_jpg(lines), points=_jpg(pts), coverage=_jpg(cov))


# ---------------------------------------------------------------------------------------------- run
def available_methods(methods):
    from .. import learned
    ok = [m for m in methods if m == "sift" or learned.torch_available()]
    skipped = [m for m in methods if m not in ok]
    if skipped:
        log("  skipped (PyTorch / LightGlue not installed): %s" % ", ".join(skipped))
    return ok


def run(src, pairs=5, methods=ALL_METHODS, accuracy="high", max_features=4096, out=None, pair_list=None, title=None):
    imgs, default_out = _inputs(src)
    out = out or default_out
    methods = available_methods([m for m in methods if m in ALL_METHODS])
    pl = pair_list or choose_pairs(imgs, pairs)
    log("Feature benchmark: %d pairs x %d methods (%s), accuracy %s, up to %d features"
        % (len(pl), len(methods), ", ".join(methods), accuracy, max_features))
    cache_img = {}

    def img(i):
        if i not in cache_img:
            cache_img[i] = read_image(imgs[i].path)
        return cache_img[i]
    res, failed = [], {}                       # one dict per (pair, method); method -> error
    for method in methods:
        try:
            res += _run_method(method, pl, imgs, img, accuracy, max_features)
        except Exception as ex:                  # e.g. GPU out of memory: report it, keep the other methods
            failed[method] = str(ex).strip().splitlines()[-1][:200] if str(ex).strip() else type(ex).__name__
            log("  %-10s FAILED: %s" % (method, failed[method]))
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    summary = _summarise(res, [m for m in methods if m not in failed], len(pl))
    _write_html(out, res, summary, pl, imgs, src, accuracy, max_features, title, failed)
    log("Feature benchmark report: %s" % out)
    best = summary[0]["method"] if summary else None
    if best:
        log("  recommended: %s" % LABEL[best])
    return out, summary


def _run_method(method, pl, imgs, img, accuracy, max_features):
    res = []
    if True:
        warm = np.full((480, 640, 3), 127, np.uint8)
        cv2.circle(warm, (320, 240), 80, (20, 20, 20), -1)
        _extract(method, warm, accuracy, max_features)            # model load / GPU warm-up, not timed
        feats = {}
        for (a, b) in pl:
            for i in (a, b):
                if i not in feats:
                    _sync(); t = time.time()
                    feats[i] = _extract(method, img(i), accuracy, max_features)
                    _sync(); feats[i] = feats[i] + (time.time() - t,)
            A, B = imgs[a], imgs[b]
            (xy0, d0, te0), (xy1, d1, te1) = feats[a], feats[b]
            s0 = (img(a).shape[1], img(a).shape[0]); s1 = (img(b).shape[1], img(b).shape[0])
            _sync(); t = time.time()
            m = _match(method, (xy0, d0), (xy1, d1), s0, s1, max_features)
            _sync(); tm = time.time() - t
            th = max(1.0, max(s0) / 1600)
            inl = np.zeros(len(m), bool)
            if len(m) >= 8:
                F, mask = cv2.findFundamentalMat(xy0[m[:, 0]], xy1[m[:, 1]], cv2.USAC_MAGSAC, th, 0.9999, 10000)
                if F is not None and mask is not None:
                    inl = mask.ravel().astype(bool)
            r = dict(pair=(a, b), method=method, kp0=len(xy0), kp1=len(xy1), matches=int(len(m)),
                     inliers=int(inl.sum()), t_extract=(te0 + te1) / 2, t_match=tm,
                     coverage=_coverage(xy0[m[inl, 0]], *s0) if inl.any() else 0.0)
            r["ok"] = r["inliers"] >= MIN_INLIERS
            if A.cam is not None and B.cam is not None and inl.any():
                e = _epipolar_px(A, B, xy0[m[inl, 0]], xy1[m[inl, 1]])
                r["epi_med"] = float(np.median(e))
                r["correct"] = float(np.mean(e < 2 * th))
            r["img"] = _draw(img(a), img(b), xy0[m[:, 0]] if len(m) else np.zeros((0, 2)),
                             xy1[m[:, 1]] if len(m) else np.zeros((0, 2)), inl)
            res.append(r)
            log("  %-10s %s / %s: %5d matches, %5d verified, coverage %3.0f%%%s"
                % (method, A.label, B.label, r["matches"], r["inliers"], 100 * r["coverage"],
                   ", %.0f%% consistent with the alignment" % (100 * r["correct"]) if "correct" in r else ""))
    return res


def _summarise(res, methods, n_pairs):
    S = []
    for m in methods:
        rr = [r for r in res if r["method"] == m]
        if not rr:
            continue
        ok = [r for r in rr if r["ok"]]
        s = dict(method=m, pairs_ok=len(ok), pairs=len(rr),
                 inliers=float(np.median([r["inliers"] for r in rr])),
                 ratio=float(np.median([r["inliers"] / r["matches"] for r in rr if r["matches"]] or [0])),
                 coverage=float(np.median([r["coverage"] for r in rr])),
                 t_extract=float(np.mean([r["t_extract"] for r in rr])), t_match=float(np.mean([r["t_match"] for r in rr])))
        corr = [r["correct"] for r in rr if "correct" in r]
        if corr:
            s["correct"] = float(np.median(corr))
            s["epi_med"] = float(np.median([r["epi_med"] for r in rr if "epi_med" in r]))
        S.append(s)
    if not S:
        return S
    top_inl = max(s["inliers"] for s in S) or 1
    for s in S:
        # pairs that align at all come first; then spread and number of verified matches; geometric
        # correctness (when the project is aligned) scales it; speed only breaks near-ties
        s["score"] = (s["pairs_ok"] / max(s["pairs"], 1)) * (0.55 * s["coverage"] / max(max(t["coverage"] for t in S), 1e-6)
                                                           + 0.45 * s["inliers"] / top_inl) * s.get("correct", 1.0)
        s["score"] -= 0.01 * (s["t_extract"] + s["t_match"]) / max(max(t["t_extract"] + t["t_match"] for t in S), 1e-6)
    S.sort(key=lambda s: -s["score"])
    return S


COLOR = dict(sift="#d9822b", superpoint="#3b7dd8", aliked="#2e9d4d", disk="#8e5bd0")


def _svg_hbars(title, S, key, fmt, unit_max=None, lower_better=False, note=""):
    """Small horizontal bar chart, one bar per method."""
    e = html.escape
    vals = [(s["method"], s.get(key)) for s in S]
    vmax = unit_max or max([v for _, v in vals if v is not None] or [1]) or 1
    W, row, lw, bw = 400, 26, 160, 170
    H = 30 + row * len(vals)
    out = ["<svg class='chart' viewBox='0 0 %d %d' width='%d' height='%d'><text x='0' y='16' class='ct'>%s</text>"
           % (W, H, W, H, e(title))]
    for k, (m, v) in enumerate(vals):
        y = 28 + k * row
        out.append("<text x='0' y='%d' class='cl'>%s</text>" % (y + 13, e(LABEL[m])))
        out.append("<rect x='%d' y='%d' width='%d' height='16' fill='#eceef2' rx='3'/>" % (lw, y, bw))
        if v is None:
            out.append("<text x='%d' y='%d' class='cv'>&ndash;</text>" % (lw + 6, y + 13))
            continue
        out.append("<rect x='%d' y='%d' width='%.1f' height='16' fill='%s' rx='3'><title>%s: %s</title></rect>"
                   % (lw, y, max(1.5, bw * min(v, vmax) / vmax), COLOR.get(m, "#888"), e(LABEL[m]), fmt(v)))
        out.append("<text x='%d' y='%d' class='cv'>%s</text>" % (lw + bw + 6, y + 13, fmt(v)))
    out.append("</svg>")
    return "<div class='cbox'>%s<div class='cn'>%s%s</div></div>" % (
        "".join(out), "lower is better. " if lower_better else "", e(note))


def _svg_pairs(title, res, S, pl, imgs, key, fmt, vmax=None):
    """Grouped vertical bars: one group per image pair, one bar per method."""
    e = html.escape
    ms = [s["method"] for s in S]
    get = lambda a, b, m: next((r.get(key) for r in res if r["pair"] == (a, b) and r["method"] == m), None)
    allv = [get(a, b, m) for (a, b) in pl for m in ms]
    top = vmax or max([v for v in allv if v is not None] or [1]) or 1
    gw = max(70, 22 * len(ms) + 18); bw = (gw - 18) / max(len(ms), 1)
    W = 40 + gw * len(pl); H = 220; base = 170; ph = 140
    out = ["<svg class='chart' viewBox='0 0 %d %d' width='%d' height='%d'><text x='0' y='14' class='ct'>%s</text>"
           % (W, H, W, H, e(title))]
    for t in (0, 0.5, 1):
        y = base - ph * t
        out.append("<line x1='36' x2='%d' y1='%.1f' y2='%.1f' stroke='#e3e6eb'/><text x='32' y='%.1f' class='ax' text-anchor='end'>%s</text>"
                   % (W, y, y, y + 4, fmt(top * t)))
    for g, (a, b) in enumerate(pl):
        x0 = 40 + g * gw + 9
        for k, m in enumerate(ms):
            v = get(a, b, m)
            if v is None:
                continue
            h = max(1.0, ph * min(v, top) / top)
            out.append("<rect x='%.1f' y='%.1f' width='%.1f' height='%.1f' fill='%s'><title>%s, %s / %s: %s</title></rect>"
                       % (x0 + k * bw, base - h, bw - 2, h, COLOR.get(m, "#888"), e(LABEL[m]),
                          e(imgs[a].label), e(imgs[b].label), fmt(v)))
        out.append("<text x='%.1f' y='%d' class='ax' text-anchor='middle'>pair %d</text>" % (x0 + (gw - 18) / 2, base + 16, g + 1))
        out.append("<text x='%.1f' y='%d' class='ax2' text-anchor='middle'>%d apart</text>" % (x0 + (gw - 18) / 2, base + 30, b - a))
    out.append("</svg>")
    return "<div class='cbox wide'>%s</div>" % "".join(out)


def _legend(S):
    e = html.escape
    return "<div class='legend'>%s</div>" % "".join(
        "<span><i style='background:%s'></i>%s</span>" % (COLOR.get(s["method"], "#888"), e(LABEL[s["method"]])) for s in S)


def _write_html(out, res, S, pl, imgs, src, accuracy, max_features, title, failed=None):
    e = html.escape
    has_gt = any("correct" in s for s in S)
    rows = []
    for k, s in enumerate(S):
        rows.append("<tr%s><td><i class='dot' style='background:%s'></i>%s%s</td><td>%d / %d</td><td>%d</td><td>%.0f%%</td><td>%.0f%%</td>%s<td>%.2f s</td><td>%.2f s</td></tr>"
                    % (" class='best'" if k == 0 else "", COLOR.get(s["method"], "#888"), e(LABEL[s["method"]]), " &#9733;" if k == 0 else "",
                       s["pairs_ok"], s["pairs"], s["inliers"], 100 * s["ratio"], 100 * s["coverage"],
                       ("<td>%.0f%%</td><td>%.2f px</td>" % (100 * s["correct"], s["epi_med"]) if "correct" in s else "<td>&ndash;</td><td>&ndash;</td>")
                       if has_gt else "", s["t_extract"], s["t_match"]))
    best = S[0] if S else None
    why = ""
    if best:
        why = ("<b>%s</b> matched %d of %d pairs with a median of %d verified matches covering %.0f%% of the image%s."
               % (e(LABEL[best["method"]]), best["pairs_ok"], best["pairs"], best["inliers"], 100 * best["coverage"],
                  ("; %.0f%% of them agree with the project's alignment" % (100 * best["correct"])) if "correct" in best else ""))
        if len(S) > 1:
            fastest = min(S, key=lambda s: s["t_extract"] + s["t_match"])
            if fastest is not best:
                why += " Fastest: %s (%.1f s per image vs %.1f s)." % (e(LABEL[fastest["method"]]),
                                                                    fastest["t_extract"] + fastest["t_match"],
                                                                    best["t_extract"] + best["t_match"])
    for m, err in (failed or {}).items():
        rows.append("<tr class='failrow'><td>%s</td><td colspan='%d'>failed: %s</td></tr>"
                    % (e(LABEL[m]), 8 if has_gt else 6, e(err)))

    # ---- charts
    for s in S:
        s["t_total"] = s["t_extract"] + s["t_match"]
        s["ok_share"] = s["pairs_ok"] / max(s["pairs"], 1)
    pct = lambda v: "%.0f%%" % (100 * v)
    num = lambda v: "%d" % round(v)
    sec = lambda v: "%.2f s" % v
    summary = [
        _svg_hbars("Pairs matched", S, "ok_share", lambda v: "%d%%" % round(100 * v), unit_max=1.0,
                   note="share of pairs with at least %d verified matches." % MIN_INLIERS),
        _svg_hbars("Median verified matches", S, "inliers", num, note="more matches = stronger tie points."),
        _svg_hbars("Coverage", S, "coverage", pct, unit_max=1.0, note="spread over the image; better geometry."),
    ]
    if has_gt:
        summary.append(_svg_hbars("Consistent with alignment", S, "correct", pct, unit_max=1.0,
                                  note="checked against the project's camera poses."))
    summary.append(_svg_hbars("Time per image", S, "t_total", sec, lower_better=True, note="extract + match."))
    per_pair = [_svg_pairs("Verified matches per pair", res, S, pl, imgs, "inliers", num),
                _svg_pairs("Coverage per pair", res, S, pl, imgs, "coverage", pct, vmax=1.0)]
    if has_gt:
        per_pair.append(_svg_pairs("Consistent with alignment per pair", res, S, pl, imgs, "correct", pct, vmax=1.0))

    # ---- per-pair image viewer
    sections = []
    for g, (a, b) in enumerate(pl):
        tabs, figs = [], []
        for k, s in enumerate(S):
            r = next((x for x in res if x["pair"] == (a, b) and x["method"] == s["method"]), None)
            if r is None:
                continue
            stat = "%d verified of %d matches &middot; coverage %.0f%%%s &middot; %.2f s + %.2f s" % (
                r["inliers"], r["matches"], 100 * r["coverage"],
                (" &middot; %.0f%% consistent with alignment" % (100 * r["correct"])) if "correct" in r else "",
                r["t_extract"], r["t_match"])
            im = r["img"] if isinstance(r["img"], dict) else dict(lines=r["img"], points=r["img"], coverage=r["img"])
            m = s["method"]
            tabs.append("<button class='tab%s' data-m='%s' style='--c:%s'>%s <small>%d</small></button>"
                        % (" on" if k == 0 else "", m, COLOR.get(m, "#888"), e(LABEL[m]), r["inliers"]))
            figs.append("<figure class='%s%s' data-m='%s'><figcaption><b style='color:%s'>%s</b> &nbsp; %s</figcaption>%s</figure>"
                        % ("fail" if not r["ok"] else "", " on" if k == 0 else "", m, COLOR.get(m, "#888"), e(LABEL[m]), stat,
                           "".join("<img class='v-%s' loading='lazy' src='data:image/jpeg;base64,%s'>" % (v, im[v])
                                   for v in ("lines", "points", "coverage"))))
        sections.append("<section class='pair'><h3>Pair %d: %s &harr; %s <span>(%d apart in capture order)</span></h3>"
                        "<div class='tabs'>%s</div><div class='figs'>%s</div></section>"
                        % (g + 1, e(imgs[a].label), e(imgs[b].label), b - a, "".join(tabs), "".join(figs)))

    doc = """<!DOCTYPE html><html><head><meta charset='utf-8'><title>%(title)s</title><style>
body{font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;background:#f6f7f9;color:#1d2330}
header{padding:28px 36px 8px} h1{font-size:22px;margin:0 0 4px} .sub{color:#5b6475} h2{font-size:17px;margin:30px 0 6px}
main{padding:0 36px 40px} table{border-collapse:collapse;background:#fff;margin:16px 0;box-shadow:0 1px 2px #0001}
th,td{padding:7px 12px;border-bottom:1px solid #e6e8ec;text-align:right} th:first-child,td:first-child{text-align:left}
th{background:#eef0f4;font-weight:600;font-size:12.5px} tr.best td{background:#e9f6ec;font-weight:600}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%%;margin-right:7px}
.rec{background:#fff;border-left:4px solid #2e9d4d;padding:12px 16px;margin:12px 0;max-width:980px;box-shadow:0 1px 2px #0001}
.charts{display:flex;flex-wrap:wrap;gap:14px} .cbox{background:#fff;padding:12px 14px;box-shadow:0 1px 2px #0001}
.cbox.wide{overflow-x:auto} .cn{font-size:11.5px;color:#6b7383;max-width:400px}
.chart .ct{font-size:13px;font-weight:600;fill:#1d2330} .chart .cl{font-size:12px;fill:#3a4252} .chart .cv{font-size:11.5px;fill:#3a4252}
.chart .ax{font-size:10.5px;fill:#6b7383} .chart .ax2{font-size:9.5px;fill:#9aa1ad}
.legend{display:flex;gap:16px;margin:8px 0;font-size:12.5px} .legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.bar{position:sticky;top:0;z-index:5;background:#f6f7f9ee;padding:10px 0;display:flex;gap:18px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #e3e6eb}
.seg{display:inline-flex;border:1px solid #c9ced8;border-radius:6px;overflow:hidden} .seg button{border:0;background:#fff;padding:5px 12px;cursor:pointer;font:inherit}
.seg button.on{background:#1d2330;color:#fff} .vk{font-size:12.5px;color:#5b6475}
section.pair{margin-top:26px} h3{font-size:15px;margin:0 0 8px} h3 span{color:#5b6475;font-weight:400}
.tabs{display:flex;gap:6px;margin-bottom:8px} .tab{border:1px solid #c9ced8;border-bottom:3px solid var(--c);background:#fff;padding:5px 12px;border-radius:5px;cursor:pointer;font:inherit}
.tab.on{background:var(--c);color:#fff} .tab small{opacity:.75;margin-left:4px}
figure{margin:0;background:#fff;box-shadow:0 1px 2px #0001;display:none;max-width:1150px} figure.on{display:block} figure img{width:100%%;display:none}
body.v-lines figure img.v-lines, body.v-points figure img.v-points, body.v-coverage figure img.v-coverage{display:block}
body.all .tabs{display:none} body.all .figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px} body.all figure{display:block;max-width:none}
figcaption{padding:7px 10px;font-size:12.5px;color:#3a4252} tr.failrow td{color:#b3261e;text-align:left} figure.fail figcaption{color:#b3261e}
.key{color:#5b6475;font-size:12.5px;max-width:980px}
</style></head><body class='v-points'><header><h1>%(title)s</h1><div class='sub'>%(src)s &middot; %(npairs)d image pairs &middot; accuracy %(acc)s &middot; up to %(maxf)d features &middot; %(when)s</div></header><main>
<div class='rec'>Recommended for this project: %(why)s</div>
<h2>Overview</h2>%(legend)s<div class='charts'>%(summary)s</div>
<h2>Per image pair</h2><p class='key'>Pairs further apart in capture order have less overlap and are harder; a method that keeps its bars up on the right-hand pairs copes better with wide baselines.</p>
<div class='charts'>%(perpair)s</div>
<h2>Numbers</h2>
<table><tr><th>Method</th><th>Pairs matched</th><th>Median verified matches</th><th>Inlier ratio</th><th>Coverage</th>%(gth)s<th>Extract / image</th><th>Match / pair</th></tr>%(rows)s</table>
<p class='key'>Coverage: share of a 10&times;10 grid over the first image holding verified matches. %(gtnote)s A pair counts as matched with at least %(mini)d verified matches. Times include model warm-up only once.</p>
<h2>Matches on the photos</h2>
<div class='bar'><span class='seg' id='view'><button data-v='points' class='on'>Points</button><button data-v='lines'>Lines</button><button data-v='coverage'>Coverage</button></span>
<span class='seg' id='lay'><button data-l='one' class='on'>One method</button><button data-l='all'>All methods</button></span>
<span class='vk' id='vk'></span></div>
%(sections)s</main>
<script>
var KEY={points:"Each verified match is a dot with the same colour in both photos (colour follows its position in the left photo), so you can see what was tied together.",
lines:"Green: verified matches (MAGSAC++), a sample. Red: rejected matches.",
coverage:"Brighter cells hold more verified matches; empty cells have none. Even coverage gives a more stable alignment."};
function seg(id,f){var s=document.getElementById(id);s.addEventListener('click',function(ev){var b=ev.target.closest('button');if(!b)return;
s.querySelectorAll('button').forEach(function(x){x.classList.toggle('on',x===b)});f(b)})}
seg('view',function(b){document.body.classList.remove('v-points','v-lines','v-coverage');document.body.classList.add('v-'+b.dataset.v);document.getElementById('vk').textContent=KEY[b.dataset.v]});
seg('lay',function(b){document.body.classList.toggle('all',b.dataset.l==='all')});
document.getElementById('vk').textContent=KEY.points;
document.querySelectorAll('section.pair').forEach(function(sec){sec.querySelector('.tabs').addEventListener('click',function(ev){
var b=ev.target.closest('.tab');if(!b)return;var m=b.dataset.m;
document.querySelectorAll('section.pair').forEach(function(s2){s2.querySelectorAll('.tab,figure').forEach(function(x){x.classList.toggle('on',x.dataset.m===m)})})})});
</script></body></html>""" % dict(
        title=e(title or "Feature method benchmark"), src=e(src), npairs=len(pl), acc=e(str(accuracy)), maxf=max_features,
        when=time.strftime("%Y-%m-%d %H:%M"), why=why or "no method matched any pair", rows="".join(rows),
        legend=_legend(S), summary="".join(summary), perpair="".join(per_pair),
        gth="<th>Consistent with alignment</th><th>Epipolar error</th>" if has_gt else "",
        gtnote=("Consistent with alignment: share of verified matches within %s of the epipolar line given by the project's camera poses."
                % "2&times; the verification threshold") if has_gt else "", mini=MIN_INLIERS, sections="".join(sections))
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="shapekit bench-features", description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="project folder (uses its photos and, if aligned, its camera poses) or an image folder")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--accuracy", default="high", choices=["highest", "high", "medium", "low", "lowest"])
    ap.add_argument("--max-features", type=int, default=4096)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.src, a.pairs, [m.strip() for m in a.methods.split(",") if m.strip()], a.accuracy, a.max_features, a.out)


if __name__ == "__main__":
    main()
