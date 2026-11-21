"""Feature detection, pair preselection, matching, geometric verification and track building.

Metashape mapping:
  accuracy  (Highest/High/Medium/Low/Lowest) -> image scale 2 / 1 / 0.5 / 0.25 / 0.125 for detection
  key point limit                            -> max_features
  generic preselection                       -> global VLAD descriptor similarity (top-k candidates)
  sequential preselection                    -> neighbours in capture order (video frames)
"""
import os
import itertools
import numpy as np
import cv2

from .util import log, read_image, UnionFind

ACCURACY = {"highest": 2.0, "high": 1.0, "medium": 0.5, "low": 0.25, "lowest": 0.125}
# Learned features: longest image side the network sees. Video frames (1920 px) are unaffected; 12+ MP photos
# would need several GB of GPU memory at native size (and can take the whole process down), so they are
# processed at this size -- keypoints are still returned in full-resolution pixel coordinates.
LEARNED_MAX_SIDE = int(os.environ.get("SHAPEKIT_MAX_SIDE", "2048"))


def _feat_path(chunk, i):
    return chunk.path("features", "%05d.npz" % i)


def feature_method(chunk):
    p = os.path.join(chunk.dir, "features", "meta.json")
    if os.path.exists(p):
        import json
        return json.load(open(p)).get("method", "sift")
    return "sift"


def detect_features(chunk, accuracy="high", max_features=10000, force=False, masks_dir=None, method="sift"):
    """method: sift | superpoint | aliked | disk  (learned ones need PyTorch + LightGlue, use the GPU)."""
    import json
    scale = ACCURACY[accuracy] if isinstance(accuracy, str) else float(accuracy)
    if method != "sift":
        from . import learned
        if not learned.torch_available():
            log("  PyTorch/LightGlue not installed -> falling back to SIFT (run `uv sync` to enable %s)" % method)
            method = "sift"
    meta_p = chunk.path("features", "meta.json")
    if os.path.exists(meta_p) and json.load(open(meta_p)).get("method") != method:
        force = True                      # switching method invalidates features and matches
        for fn in ("matches.npz", "matches_partial.npz", "align_checkpoint.npz"):
            mp = os.path.join(chunk.dir, fn)
            if os.path.exists(mp):
                os.remove(mp)
    if method == "sift":
        sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
    else:
        from . import learned
        lf = learned.get(method, max_features, None)
        log("  %s + LightGlue on %s" % (method, learned.describe_device()))
        big = sorted({(s.width, s.height) for s in chunk.sensors if max(s.width, s.height) * min(scale, 1) > LEARNED_MAX_SIDE})
        if big:
            log("  photos larger than %d px (%s) are analysed at %d px on their longest side (GPU memory); "
                "keypoints keep full-resolution coordinates" % (LEARNED_MAX_SIDE, ", ".join("%dx%d" % b for b in big),
                                                                 LEARNED_MAX_SIDE))
    cached = sum(os.path.exists(_feat_path(chunk, i)) for i in range(len(chunk.cameras))) if not force else 0
    if cached:
        log("  features already computed for %d/%d photos - reused" % (cached, len(chunk.cameras)))
    for i, cam in enumerate(chunk.cameras):
        p = _feat_path(chunk, i)
        if os.path.exists(p) and not force:
            continue
        if method != "sift":
            img = read_image(chunk.abspath(cam))
            side = max(img.shape[:2])
            r = min(side if scale >= 1 else int(round(side * scale)), LEARNED_MAX_SIDE)
            f = lf.extract(img, resize=None if r >= side else r)
            xy = f["xy"]
            xi = np.clip(xy.round().astype(int), 0, [img.shape[1] - 1, img.shape[0] - 1])
            rgb = img[xi[:, 1], xi[:, 0], ::-1] if len(xy) else np.zeros((0, 3), np.uint8)
            np.savez(p, xy=xy.astype(np.float32), desc=f["desc"], rgb=rgb.astype(np.uint8),
                     size=np.full(len(xy), 8.0, np.float32), score=f["score"].astype(np.float32),
                     image_size=np.array([img.shape[1], img.shape[0]], np.float32))
            if i % 20 == 0:
                log("  features %d/%d: %d keypoints" % (i + 1, len(chunk.cameras), len(xy)))
            continue
        img = read_image(chunk.abspath(cam), scale=scale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = None
        if masks_dir:
            mp = os.path.join(masks_dir, cam.label + ".png")
            if os.path.exists(mp):
                mask = cv2.resize(cv2.imread(mp, 0), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        kps, desc = sift.detectAndCompute(gray, mask)
        if desc is None or len(kps) == 0:
            np.savez(p, xy=np.zeros((0, 2), np.float32), desc=np.zeros((0, 128), np.float16),
                     rgb=np.zeros((0, 3), np.uint8), size=np.zeros(0, np.float32))
            continue
        xy = np.array([k.pt for k in kps], np.float32)
        # RootSIFT
        desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc)
        xi = np.clip(xy.round().astype(int), 0, [img.shape[1] - 1, img.shape[0] - 1])
        rgb = img[xi[:, 1], xi[:, 0], ::-1]
        xy_full = (xy + 0.5) / scale - 0.5
        np.savez(p, xy=xy_full.astype(np.float32), desc=desc.astype(np.float16), rgb=rgb.astype(np.uint8),
                 size=np.array([k.size / scale for k in kps], np.float32))
        if i % 20 == 0:
            log("  features %d/%d: %d keypoints" % (i + 1, len(chunk.cameras), len(kps)))
    json.dump(dict(method=method, accuracy=accuracy, max_features=max_features), open(meta_p, "w"))
    return method


def load_features(chunk, i):
    d = np.load(_feat_path(chunk, i))
    return d["xy"], d["desc"].astype(np.float32), d["rgb"]


# ----------------------------------------------------------------------------- preselection
def vlad_descriptors(chunk, k=48, sample=200000, seed=0):
    rng = np.random.default_rng(seed)
    descs = [load_features(chunk, i)[1] for i in range(len(chunk.cameras))]
    allsamp = np.concatenate([d[rng.choice(len(d), min(len(d), max(1, sample // len(descs))), replace=False)]
                              for d in descs if len(d)])
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    _, _, centers = cv2.kmeans(allsamp.astype(np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    D = allsamp.shape[1]
    out = np.zeros((len(descs), k * D), np.float32)
    for i, d in enumerate(descs):
        if len(d) == 0:
            continue
        dist = (d ** 2).sum(1)[:, None] - 2 * d @ centers.T + (centers ** 2).sum(1)[None]
        a = dist.argmin(1)
        v = np.zeros((k, D), np.float32)
        np.add.at(v, a, d - centers[a])
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        v = v.ravel()
        v = np.sign(v) * np.sqrt(np.abs(v))
        out[i] = v / (np.linalg.norm(v) + 1e-9)
    return out


def preselect_pairs(chunk, sequential_window=10, generic_k=12, exhaustive_below=40, retrieval="auto",
                    cross_k=6):
    """Pair preselection = sequential (capture order within each video) + retrieval.

    retrieval: 'auto' (SALAD if PyTorch is available, else VLAD) | 'salad' | 'dinov2' | 'vlad'
    generic_k: retrieval neighbours per photo outside its own sequential window (loop closures)
    cross_k:   extra retrieval neighbours per photo taken only from OTHER videos / photo sets, so separate
               recordings of the same area get connected even when within-video similarity dominates.
    """
    idx = [i for i, c in enumerate(chunk.cameras) if c.enabled]
    n = len(idx)
    pairs = set()
    if n <= exhaustive_below:
        return sorted(itertools.combinations(idx, 2))
    by_src = {}
    for i in idx:
        c = chunk.cameras[i]
        by_src.setdefault(c.source or "__photos__", []).append(i)
    for src, lst in by_src.items():
        if src == "__photos__":
            continue
        lst.sort(key=lambda i: chunk.cameras[i].frame)
        for a in range(len(lst)):
            for b in range(a + 1, min(len(lst), a + 1 + sequential_window)):
                pairs.add((min(lst[a], lst[b]), max(lst[a], lst[b])))
    n_seq = len(pairs)
    # Similarity matrices to rank candidates. 'auto' = hybrid: candidates are interleaved from SALAD (learned,
    # robust to lighting / season / large viewpoint change) and VLAD of local features (best at finding actual
    # surface overlap). Benchmark on area_d (3 videos, verified with ALIKED+LightGlue): queries with a verified
    # cross-video loop closure in top-5 -- SALAD 16/40, DINOv2 17/40, VLAD 23/40, hybrid 25/40.
    sims, kinds = [], []
    want = ["salad", "vlad"] if retrieval == "auto" else [retrieval]
    for kind in want:
        try:
            if kind == "vlad":
                V = vlad_descriptors(chunk)[idx]
            else:
                from .learned import torch_available
                if not torch_available():
                    raise RuntimeError("PyTorch not installed")
                from .learned.retrieval import global_descriptors
                V = global_descriptors(chunk, kind)[idx]
            S_ = V @ V.T
            np.fill_diagonal(S_, -np.inf)
            sims.append(S_); kinds.append(kind)
        except Exception as e:
            log("  %s retrieval unavailable (%s)" % (kind, str(e).splitlines()[0][:120]))
    if not sims:
        V = vlad_descriptors(chunk)[idx]
        S_ = V @ V.T; np.fill_diagonal(S_, -np.inf)
        sims, kinds = [S_], ["vlad"]
    kind = "+".join(kinds)

    def ranking(a):
        orders = [np.argsort(-S_[a]) for S_ in sims]
        if len(orders) == 1:
            return orders[0]
        out, seen = [], set()
        for tup in zip(*orders):
            for b in tup:
                if b not in seen:
                    seen.add(b); out.append(b)
        return np.array(out)
    pos = {}
    for src, lst in by_src.items():
        for r, i in enumerate(lst):
            pos[i] = (src, r)
    src_arr = np.array([pos[i][0] for i in idx])
    multi = len(by_src) > 1
    for a in range(n):
        order = ranking(a)
        added = 0
        for b in order:
            if added >= generic_k:
                break
            i, j = idx[a], idx[b]
            pi, pj = pos[i], pos[j]
            if pi[0] == pj[0] and pi[0] != "__photos__" and abs(pi[1] - pj[1]) <= sequential_window:
                continue            # already matched sequentially -> spend the budget on loop closures
            pairs.add((min(i, j), max(i, j)))
            added += 1
        if multi and cross_k:
            other = order[src_arr[order] != src_arr[a]][:cross_k]
            for b in other:
                i, j = idx[a], idx[b]
                pairs.add((min(i, j), max(i, j)))
    if multi:
        n_cross = sum(1 for i, j in pairs if pos[i][0] != pos[j][0])
        log("  preselection (%s): %d sequential + %d retrieval pairs, %d of them across %d videos/sets" %
            (kind, n_seq, len(pairs) - n_seq, n_cross, len(by_src)))
    else:
        log("  preselection (%s): %d sequential + %d retrieval pairs" % (kind, n_seq, len(pairs) - n_seq))
    return sorted(pairs)


# ----------------------------------------------------------------------------- matching
def match_pair(d1, d2, ratio=0.8):
    if len(d1) < 8 or len(d2) < 8:
        return np.zeros((0, 2), int)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    m12 = flann.knnMatch(d1, d2, k=2)
    m21 = flann.knnMatch(d2, d1, k=2)
    best21 = np.full(len(d2), -1)
    for m in m21:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            best21[m[0].queryIdx] = m[0].trainIdx
    out = [(m[0].queryIdx, m[0].trainIdx) for m in m12
           if len(m) == 2 and m[0].distance < ratio * m[1].distance and best21[m[0].trainIdx] == m[0].queryIdx]
    return np.array(out, int).reshape(-1, 2)


def _match_job(args):
    chunk_dir, i, j, min_inliers, th = args
    a = np.load(os.path.join(chunk_dir, "features", "%05d.npz" % i))
    b = np.load(os.path.join(chunk_dir, "features", "%05d.npz" % j))
    xy1, d1 = a["xy"], a["desc"].astype(np.float32)
    xy2, d2 = b["xy"], b["desc"].astype(np.float32)
    m = match_pair(d1, d2)
    if len(m) < min_inliers:
        return i, j, None
    F, inl = cv2.findFundamentalMat(xy1[m[:, 0]], xy2[m[:, 1]], cv2.USAC_MAGSAC, th, 0.9999, 10000)
    if F is None or inl is None:
        return i, j, None
    inl = inl.ravel().astype(bool)
    if inl.sum() < min_inliers:
        return i, j, None
    return i, j, m[inl]


def match_features(chunk, pairs, min_inliers=25, px_thresh=None, force=False, workers=None):
    out_p = chunk.path("matches.npz")
    if os.path.exists(out_p) and not force:
        log("  matches already computed - reused")
        return
    if force:
        for fn in ("matches_partial.npz", "align_checkpoint.npz"):
            if os.path.exists(os.path.join(chunk.dir, fn)):
                os.remove(os.path.join(chunk.dir, fn))
    method = feature_method(chunk)
    if method != "sift":
        return _match_learned(chunk, pairs, method, min_inliers, px_thresh, out_p)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    jobs = []
    for i, j in pairs:
        s = chunk.sensors[chunk.cameras[i].sensor]
        th = px_thresh or max(1.0, 1.0 * max(s.width, s.height) / 1600)
        jobs.append((chunk.dir, i, j, min_inliers, th))
    results = []
    if workers > 1 and len(jobs) > 20:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            for k, r in enumerate(ex.map(_match_job, jobs, chunksize=8)):
                results.append(r)
                if k % 200 == 0:
                    log("  matched %d/%d pairs" % (k + 1, len(jobs)))
    else:
        for k, jb in enumerate(jobs):
            results.append(_match_job(jb))
            if k % 100 == 0:
                log("  matched %d/%d pairs" % (k + 1, len(jobs)))
    P, M = [], []
    for i, j, mm in results:
        if mm is not None:
            P.append((i, j, len(mm))); M.append(mm)
    pairs_arr = np.array(P, int).reshape(-1, 3)
    counts = pairs_arr[:, 2]
    np.savez_compressed(out_p, pairs=pairs_arr,
                        matches=np.concatenate(M) if M else np.zeros((0, 2), int),
                        offsets=np.concatenate([[0], np.cumsum(counts)]))
    log("  %d/%d pairs verified, median %d inliers" % (len(P), len(pairs), int(np.median(counts)) if len(P) else 0))


def _verify(xy1, xy2, m, th, min_inliers):
    if len(m) < min_inliers:
        return None
    F, inl = cv2.findFundamentalMat(xy1[m[:, 0]], xy2[m[:, 1]], cv2.USAC_MAGSAC, th, 0.9999, 10000)
    if F is None or inl is None or inl.sum() < min_inliers:
        return None
    return m[inl.ravel().astype(bool)]


def _match_learned(chunk, pairs, method, min_inliers, px_thresh, out_p):
    """LightGlue on the GPU with descriptors cached on the device; MAGSAC++ verification runs on CPU threads
    in parallel. Progress is checkpointed (matches_partial.npz) so an interrupted run resumes."""
    import json
    import time as _t
    from concurrent.futures import ThreadPoolExecutor
    from . import learned
    meta = json.load(open(os.path.join(chunk.dir, "features", "meta.json")))
    sen0 = chunk.sensors[chunk.cameras[0].sensor]
    sc = ACCURACY[meta.get("accuracy", "high")]
    lf = learned.get(method, meta.get("max_features", 4096),
                     None if sc >= 1 else int(round(max(sen0.width, sen0.height) * sc)))
    part_p = os.path.join(chunk.dir, "matches_partial.npz")
    P, M = [], []
    done = set()
    if os.path.exists(part_p):
        d = np.load(part_p)
        if d["method"].item() == method:
            done = {(int(a), int(b)) for a, b in d["done"]}
            off = d["offsets"]
            P = [tuple(map(int, p)) for p in d["pairs"]]
            M = [d["matches"][off[k]:off[k + 1]] for k in range(len(P))]
            log("  resuming matching: %d/%d pairs already done" % (len(done), len(pairs)))
    todo = [p for p in pairs if p not in done]
    host = {}

    def get(i):
        if i not in host:
            if len(host) > 400:
                host.pop(next(iter(host)))
            d = np.load(_feat_path(chunk, i))
            host[i] = (d["xy"], lf.to_device(i, d["xy"], d["desc"], tuple(d["image_size"].tolist())))
        return host[i]

    def save_partial():
        counts = np.array([p[2] for p in P], int)
        tmp = part_p + ".tmp.npz"
        np.savez(tmp, method=np.array(method), done=np.array(sorted(done), int).reshape(-1, 2),
                 pairs=np.array(P, int).reshape(-1, 3), matches=np.concatenate(M) if M else np.zeros((0, 2), int),
                 offsets=np.concatenate([[0], np.cumsum(counts)]))
        os.replace(tmp, part_p)

    t0 = _t.time()
    pending = []
    with ThreadPoolExecutor(max(2, min(8, (os.cpu_count() or 4) - 1))) as ex:
        for k, (i, j) in enumerate(todo):
            xy1, g1 = get(i)
            xy2, g2 = get(j)
            m, _ = lf.match_dev(g1, g2)
            s = chunk.sensors[chunk.cameras[i].sensor]
            th = px_thresh or max(1.0, 1.0 * max(s.width, s.height) / 1600)
            pending.append(((i, j), ex.submit(_verify, xy1, xy2, m, th, min_inliers)))
            if len(pending) > 64 or k == len(todo) - 1:
                for (a, b), fut in pending:
                    mm = fut.result()
                    if mm is not None:
                        P.append((a, b, len(mm))); M.append(mm)
                    done.add((a, b))
                pending = []
            if k % 200 == 0:
                rate = (k + 1) / max(_t.time() - t0, 1e-6)
                log("  LightGlue %d/%d pairs (%.1f pairs/s, ~%.0f s left)" % (k + 1 + len(pairs) - len(todo), len(pairs),
                                                                           rate, (len(todo) - k - 1) / rate))
            if k % 2000 == 1999:
                save_partial()
    pairs_arr = np.array(P, int).reshape(-1, 3)
    counts = pairs_arr[:, 2]
    np.savez_compressed(out_p, pairs=pairs_arr, matches=np.concatenate(M) if M else np.zeros((0, 2), int),
                        offsets=np.concatenate([[0], np.cumsum(counts)]))
    if os.path.exists(part_p):
        os.remove(part_p)
    log("  %d/%d pairs verified, median %d inliers" % (len(P), len(pairs), int(np.median(counts)) if len(P) else 0))


def load_matches(chunk):
    d = np.load(chunk.path("matches.npz"))
    pairs, matches, off = d["pairs"], d["matches"], d["offsets"]
    return {(int(p[0]), int(p[1])): matches[off[k]:off[k + 1]] for k, p in enumerate(pairs)}


def build_tracks(chunk, matches):
    """Union-find keypoints into tracks. Returns per-observation arrays (img, kp, track)."""
    log("  linking matches of %d photo pairs into multi-photo tracks..." % len(matches))
    ncam = len(chunk.cameras)
    nk = np.array([len(np.load(_feat_path(chunk, i))["xy"]) for i in range(ncam)])
    off = np.concatenate([[0], np.cumsum(nk)])
    A, B = [], []
    for (i, j), m in matches.items():
        A.append(off[i] + m[:, 0]); B.append(off[j] + m[:, 1])
    if not A:
        raise RuntimeError("no matches")
    A, B = np.concatenate(A), np.concatenate(B)
    used = np.unique(np.concatenate([A, B]))
    remap = -np.ones(off[-1], int)
    remap[used] = np.arange(len(used))
    uf = UnionFind(len(used))
    roots = uf.union_many(remap[A], remap[B])
    img = np.searchsorted(off, used, side="right") - 1
    kp = used - off[img]
    _, track = np.unique(roots, return_inverse=True)
    # drop duplicate (track, image) observations -> keep first
    order = np.lexsort((kp, img, track))
    img, kp, track = img[order], kp[order], track[order]
    key = track * ncam + img
    keep = np.concatenate([[True], key[1:] != key[:-1]])
    img, kp, track = img[keep], kp[keep], track[keep]
    cnt = np.bincount(track)
    ok = cnt[track] >= 2
    img, kp, track = img[ok], kp[ok], track[ok]
    _, track = np.unique(track, return_inverse=True)
    return img, kp, track
