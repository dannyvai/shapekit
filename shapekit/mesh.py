"""Build Mesh (depth-maps based) + mesh tools (filter components, smooth, decimate, close-holes-lite).

Method: truncated signed distance fusion of filtered depth maps in a narrow band around the
surface (sparse in memory), surface extraction by marching cubes, vertex colours from the views.
"""
import os
import time
import numpy as np
import cv2
import scipy.ndimage as ndi
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

from . import camera as cm
from . import mvs
from .io import write_ply, read_ply
from .util import log, read_image

FACE_COUNT = {"high": 1.5, "medium": 3.0, "low": 6.0}   # voxel size in multiples of depth-map GSD


def build_mesh(chunk, face_count="medium", voxel_size=None, trunc_vox=4, min_weight=2, max_voxels=250e6,
               component_filter=0.02, smooth_iters=3):
    t0 = time.time()
    c, s, R = mvs.region_box(chunk)
    cams = [i for i in chunk.aligned_cameras() if mvs.load_depth(chunk, i) is not None]
    if not cams:
        raise RuntimeError("build depth maps first")
    scale = mvs.load_depth(chunk, cams[0])["K"][0, 0] / chunk.sensors[chunk.cameras[cams[0]].sensor].intr[0]
    gsd = mvs.estimate_gsd(chunk, scale)
    vox = voxel_size or gsd * FACE_COUNT[face_count]
    dims = np.ceil(s / vox).astype(int)
    while np.prod(dims.astype(float)) > max_voxels:
        vox *= 1.25; dims = np.ceil(s / vox).astype(int)
    origin = -s / 2
    log("Build Mesh: voxel %.4g, grid %s, %d cameras" % (vox, "x".join(map(str, dims)), len(cams)))
    trunc = trunc_vox * vox

    # ---- narrow band from depth-map samples
    occ = np.zeros(dims, bool)

    def band_cells(i):
        d = mvs.load_depth(chunk, i)
        D = d["depth"]
        if (D > 0).sum() == 0:
            return None
        cam = chunk.cameras[i]
        Rc, tc = cm.rodrigues_to_R(cam.rvec)[0], np.asarray(cam.tvec)
        ys, xs = np.nonzero(D[::2, ::2]); ys, xs = ys * 2, xs * 2
        z = D[ys, xs]
        Xc = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float64) @ np.linalg.inv(d["K"]).T * z[:, None]
        L = ((Xc - tc) @ Rc - c) @ R.T
        g = np.floor((L - origin) / vox).astype(int)
        ok = ((g >= 0) & (g < dims)).all(1)
        return g[ok]
    from .util import fast_mode, n_workers
    if fast_mode():
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(n_workers(8)) as ex:
            cells = ex.map(band_cells, cams)
            for g in cells:
                if g is not None:
                    occ[g[:, 0], g[:, 1], g[:, 2]] = True
    else:
        for i in cams:
            g = band_cells(i)
            if g is not None:
                occ[g[:, 0], g[:, 1], g[:, 2]] = True
    band = ndi.binary_dilation(occ, iterations=trunc_vox)
    del occ
    bi = np.flatnonzero(band)
    log("  narrow band: %d voxels (%.1f%% of grid)" % (len(bi), 100 * len(bi) / band.size))
    gx, gy, gz = np.unravel_index(bi, dims)
    Lc = (np.stack([gx, gy, gz], 1) + 0.5) * vox + origin
    W = Lc @ R + c                                     # world coords of band voxels
    del Lc, gx, gy, gz
    gpu_ok = False
    if fast_mode():
        try:
            from .learned.mvs_gpu import gpu_available
            gpu_ok = gpu_available()
        except Exception:
            gpu_ok = False
    if gpu_ok:
        tsdf, wsum, col, cw = _integrate_gpu(chunk, cams, W, trunc)
        cams_cpu = []
    else:
        tsdf = np.zeros(len(bi), np.float32)
        wsum = np.zeros(len(bi), np.float32)
        col = np.zeros((len(bi), 3), np.float32)
        cw = np.zeros(len(bi), np.float32)
        cams_cpu = cams
    for k, i in enumerate(cams_cpu):
        d = mvs.load_depth(chunk, i)
        D, K = d["depth"], d["K"]
        cam = chunk.cameras[i]
        sen = chunk.sensors[cam.sensor]
        Rc, tc = cm.rodrigues_to_R(cam.rvec)[0], np.asarray(cam.tvec)
        h, w = D.shape
        for a in range(0, len(W), 4_000_000):
            Wb = W[a:a + 4_000_000]
            Xc = Wb @ Rc.T + tc
            z = Xc[:, 2]
            front = z > 1e-6
            u = np.where(front, Xc[:, 0] / np.maximum(z, 1e-6) * K[0, 0] + K[0, 2], -1)
            v = np.where(front, Xc[:, 1] / np.maximum(z, 1e-6) * K[1, 1] + K[1, 2], -1)
            ui, vi = np.round(u).astype(np.int32), np.round(v).astype(np.int32)
            inb = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            idx = np.where(inb)[0]
            dz = D[vi[idx], ui[idx]]
            ok = dz > 0
            idx, dz = idx[ok], dz[ok]
            sdf = dz - z[idx]
            ok = sdf > -trunc
            idx, sdf = idx[ok], sdf[ok]
            val = np.minimum(1.0, sdf / trunc)
            wt = np.ones_like(val)
            tsdf[a + idx] += val * wt
            wsum[a + idx] += wt
            near = np.abs(sdf) < trunc * 0.5
            if near.any():
                if k == 0 or not hasattr(build_mesh, "_cache") or build_mesh._cache[0] != i:
                    img = read_image(chunk.abspath(cam))
                    m1, m2, _ = cm.undistort_maps(sen.calib(), sen.width, sen.height, w / sen.width)
                    build_mesh._cache = (i, cv2.remap(img, m1, m2, cv2.INTER_AREA))
                im = build_mesh._cache[1]
                j = idx[near]
                col[a + j] += im[vi[j], ui[j], ::-1]
                cw[a + j] += 1
        if k % 10 == 0:
            log("  integrated %d/%d views" % (k + 1, len(cams)))
    if hasattr(build_mesh, "_cache"):
        del build_mesh._cache
    del W
    obs = wsum >= min_weight
    vol = np.ones(dims, np.float32)
    vol.flat[bi[obs]] = tsdf[obs] / wsum[obs]
    mask = np.zeros(dims, bool)
    mask.flat[bi[obs]] = True
    colvol_idx = -np.ones(dims, np.int64)
    colvol_idx.flat[bi] = np.arange(len(bi))
    verts, faces, _, _ = marching_cubes(vol, 0.0, mask=mask, allow_degenerate=False)
    # skimage winding already gives normals pointing to the observed (positive) side
    faces = faces[:, :]
    del vol
    # drop faces generated against unobserved voxels (spurious back shells at the band boundary)
    fl = np.floor(verts).astype(int); ce = np.ceil(verts).astype(int)
    fl = np.clip(fl, 0, dims - 1); ce = np.clip(ce, 0, dims - 1)
    vok = mask[fl[:, 0], fl[:, 1], fl[:, 2]] & mask[ce[:, 0], ce[:, 1], ce[:, 2]]
    # also require the negative (behind-surface) side to be a real observation, not a far default
    faces = faces[vok[faces].all(1)]
    # vertex colours: nearest voxel with colour
    vi = np.clip(np.round(verts).astype(int), 0, dims - 1)
    ci = colvol_idx[vi[:, 0], vi[:, 1], vi[:, 2]]
    vcol = np.full((len(verts), 3), 128.0)
    okc = (ci >= 0)
    okc[okc] &= cw[ci[okc]] > 0
    vcol[okc] = col[ci[okc]] / cw[ci[okc], None]
    Lv = (verts + 0.5) * vox + origin
    V = Lv @ R + c
    log("  marching cubes: %d vertices, %d faces" % (len(V), len(faces)))
    V, faces, vcol = remove_small_components(V, faces, vcol, component_filter)
    if smooth_iters:
        V = taubin_smooth(V, faces, smooth_iters)
    write_ply(chunk.path("mesh.ply"), V, faces=faces, colors=vcol.clip(0, 255).astype(np.uint8))
    chunk.stats["mesh"] = dict(vertices=int(len(V)), faces=int(len(faces)), voxel=float(vox))
    chunk.add_log("Build Mesh", time.time() - t0, faces=int(len(faces)), voxel=float(vox))
    log("Mesh: %d vertices, %d faces" % (len(V), len(faces)))
    return V, faces, vcol


def _integrate_gpu(chunk, cams, W, trunc):
    """TSDF + colour integration on CUDA. Same arithmetic as the numpy loop: float64 geometry, float32
    accumulators updated as float32(float32 + float64) like numpy's in-place add."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    dev = torch.device("cuda")
    Wt = torch.from_numpy(W).to(dev)
    n = len(W)
    tsdf = torch.zeros(n, dtype=torch.float32, device=dev)
    wsum = torch.zeros(n, dtype=torch.float32, device=dev)
    col = torch.zeros((n, 3), dtype=torch.float32, device=dev)
    cw = torch.zeros(n, dtype=torch.float32, device=dev)

    def load(i):
        d = mvs.load_depth(chunk, i)
        cam = chunk.cameras[i]
        sen = chunk.sensors[cam.sensor]
        D = d["depth"]
        h, w = D.shape
        img = read_image(chunk.abspath(cam))
        m1, m2, _ = cm.undistort_maps(sen.calib(), sen.width, sen.height, w / sen.width)
        im = cv2.remap(img, m1, m2, cv2.INTER_AREA)
        return D, d["K"], cm.rodrigues_to_R(cam.rvec)[0], np.asarray(cam.tvec, np.float64), im
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(load, i) for i in cams[:6]]
        for k, i in enumerate(cams):
            if k + 6 < len(cams):
                futs.append(ex.submit(load, cams[k + 6]))
            D, K, Rc, tc, im = futs[k].result()
            futs[k] = None
            h, w = D.shape
            Dt = torch.from_numpy(D).to(dev)
            imt = torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1])).to(dev)
            Xc = Wt @ torch.from_numpy(Rc.T.copy()).to(dev) + torch.from_numpy(tc).to(dev)
            z = Xc[:, 2]
            front = z > 1e-6
            zc = torch.clamp_min(z, 1e-6)
            u = torch.where(front, Xc[:, 0] / zc * float(K[0, 0]) + float(K[0, 2]), torch.full_like(z, -1.0))
            v = torch.where(front, Xc[:, 1] / zc * float(K[1, 1]) + float(K[1, 2]), torch.full_like(z, -1.0))
            ui = torch.round(u.clamp(-2 ** 30, 2 ** 30)).to(torch.int64)
            vi = torch.round(v.clamp(-2 ** 30, 2 ** 30)).to(torch.int64)
            inb = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            idx = torch.nonzero(inb)[:, 0]
            dz = Dt[vi[idx], ui[idx]]
            ok = dz > 0
            idx, dz = idx[ok], dz[ok]
            sdf = dz.double() - z[idx]
            ok = sdf > -trunc
            idx, sdf = idx[ok], sdf[ok]
            val = torch.clamp(sdf / trunc, max=1.0)
            tsdf[idx] = (tsdf[idx].double() + val).float()
            wsum[idx] = (wsum[idx].double() + 1.0).float()
            near = sdf.abs() < trunc * 0.5
            j = idx[near]
            col[j] += imt[vi[j], ui[j]].float()
            cw[j] += 1
            del Xc, z, u, v, ui, vi, inb, idx
            if k % 10 == 0:
                log("  integrated %d/%d views (GPU)" % (k + 1, len(cams)))
    out = tsdf.cpu().numpy(), wsum.cpu().numpy(), col.cpu().numpy(), cw.cpu().numpy()
    del Wt, tsdf, wsum, col, cw
    torch.cuda.empty_cache()
    return out


def load_mesh(chunk, name="mesh.ply"):
    m = read_ply(chunk.path(name))
    return m["verts"], m["faces"], m.get("colors")


def vertex_adjacency(nv, faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    return A


def remove_small_components(V, F, C=None, frac=0.02):
    A = vertex_adjacency(len(V), F)
    n, lab = connected_components(A, directed=False)
    fl = lab[F[:, 0]]
    cnt = np.bincount(fl, minlength=n)
    keep = cnt[fl] >= frac * cnt.max()
    F = F[keep]
    used = np.unique(F)
    remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used], remap[F], (C[used] if C is not None else None)


def taubin_smooth(V, F, iters=3, lam=0.5, mu=-0.53):
    A = vertex_adjacency(len(V), F)
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    Dinv = sp.diags(1 / deg)
    L = Dinv @ A
    V = V.copy()
    for _ in range(iters):
        V = V + lam * (L @ V - V)
        V = V + mu * (L @ V - V)
    return V


def decimate_clustering(V, F, target_faces, C=None):
    """Fast vertex-clustering decimation to roughly target_faces."""
    if len(F) <= target_faces:
        return V, F, C
    ext = V.max(0) - V.min(0)
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
    cell = np.sqrt(area / target_faces) * 1.1
    for _ in range(12):
        key = np.floor((V - V.min(0)) / cell).astype(np.int64)
        dims = key.max(0) + 1
        lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
        uniq, inv = np.unique(lin, return_inverse=True)
        NF = inv[F]
        good = (NF[:, 0] != NF[:, 1]) & (NF[:, 1] != NF[:, 2]) & (NF[:, 0] != NF[:, 2])
        NF = NF[good]
        NF = np.unique(np.sort(NF, axis=1), axis=0) if False else NF
        if len(NF) <= target_faces * 1.1:
            break
        cell *= np.sqrt(len(NF) / target_faces)
    cnt = np.bincount(inv)
    NV = np.stack([np.bincount(inv, V[:, k]) for k in range(3)], 1) / cnt[:, None]
    NC = None
    if C is not None:
        NC = np.stack([np.bincount(inv, C[:, k].astype(float)) for k in range(3)], 1) / cnt[:, None]
    used = np.unique(NF)
    remap = -np.ones(len(NV), int); remap[used] = np.arange(len(used))
    return NV[used], remap[NF], (NC[used] if NC is not None else None)


def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-20)[:, None], 0.5 * a


# ----------------------------------------------------------------------------- close holes
def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])     # area weighted
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-20)


def boundary_loops(F):
    """Closed boundary loops (vertex index lists, oriented like the faces' edges)."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
    nv = int(F.max()) + 1
    key = e[:, 0] * nv + e[:, 1]
    rkey = e[:, 1] * nv + e[:, 0]
    bnd = e[~np.isin(key, rkey)]
    nxt = {}
    multi = set()
    for a, b in bnd:
        if a in nxt:
            multi.add(a)
        nxt[a] = b
    loops, seen = [], set()
    for a0 in list(nxt):
        if a0 in seen or a0 in multi:
            continue
        loop, a = [a0], nxt[a0]
        ok = True
        while a != a0:
            if a in seen or a in multi or a not in nxt or len(loop) > len(nxt):
                ok = False
                break
            loop.append(a); a = nxt[a]
        seen.update(loop)
        if ok and len(loop) >= 3:
            loops.append(loop)
    return loops


def close_holes(V, F, C=None, max_edges=200, smooth_iters=60):
    """Fill boundary loops of up to max_edges edges with a smooth membrane (concentric rings + Laplacian
    relaxation with the hole border fixed). The largest loop is never filled (outer border of an open scan)."""
    loops = boundary_loops(F)
    if not loops:
        return V, F, C, 0
    longest = max(range(len(loops)), key=lambda k: len(loops[k]))
    V = [V]; F = [F]; Cn = [C] if C is not None else None
    nv = len(V[0]); new_ids = []
    nb_new = {}
    filled = 0
    for li, loop in enumerate(loops):
        n = len(loop)
        if li == longest or n > max_edges:
            continue
        B = V[0][loop]
        c = B.mean(0)
        m = int(np.clip(round(n / 12), 1, 8))           # number of rings (incl. the border)
        rings = [np.array(loop)]
        newv = []
        for j in range(1, m):
            P = B + (c - B) * (j / m)
            ids = np.arange(nv, nv + n); nv += n
            newv.append(P); rings.append(ids)
        cid = nv; nv += 1
        newv.append(c[None])
        faces = []
        for j in range(len(rings) - 1):
            O, I = rings[j], rings[j + 1]
            for i in range(n):
                i2 = (i + 1) % n
                faces.append((O[i2], O[i], I[i])); faces.append((O[i2], I[i], I[i2]))
        I = rings[-1]
        for i in range(n):
            faces.append((I[(i + 1) % n], I[i], cid))
        Vn = np.concatenate(newv)
        V.append(Vn); F.append(np.array(faces, np.int64))
        if Cn is not None:
            Cn.append(np.repeat(np.asarray(C[loop], float).mean(0, keepdims=True), len(Vn), 0))
        new_ids.append(np.arange(cid + 1 - len(Vn), cid + 1))
        filled += 1
    if not filled:
        return V[0], F[0], C, 0
    V = np.concatenate(V); F = np.concatenate([f.astype(np.int64) for f in F])
    C = np.concatenate(Cn).round().clip(0, 255).astype(np.uint8) if Cn is not None else None
    new = np.concatenate(new_ids)
    A = vertex_adjacency(len(V), F).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    isnew = np.zeros(len(V), bool); isnew[new] = True
    for _ in range(smooth_iters):                       # harmonic membrane, border fixed
        avg = (A @ V) / np.maximum(deg, 1)[:, None]
        V[isnew] = avg[isnew]
        if C is not None:
            Cf = C.astype(float); Cf[isnew] = ((A @ Cf) / np.maximum(deg, 1)[:, None])[isnew]
            C = Cf.round().clip(0, 255).astype(np.uint8)
    return V, F, C, filled


def close_holes_chunk(chunk, max_edges=200):
    import time
    t0 = time.time()
    V, F, C = load_mesh(chunk)
    V2, F2, C2, n = close_holes(V, F.astype(np.int64), C, max_edges)
    write_ply(chunk.path("mesh.ply"), V2, faces=F2, colors=C2)
    chunk.stats["mesh"] = dict(chunk.stats.get("mesh", {}), vertices=int(len(V2)), faces=int(len(F2)))
    chunk.stats.pop("texture", None)
    chunk.add_log("Close Holes", time.time() - t0, filled=n)
    log("Close Holes: filled %d holes (+%d faces)" % (n, len(F2) - len(F)))
    return n


# ----------------------------------------------------------------------------- photometric refinement
def _sample_bilinear(img, x, y):
    """img (h,w) float32; x, y float arrays (any shape) -> values, valid."""
    h, w = img.shape
    shp = x.shape
    x = x.reshape(-1).astype(np.float32); y = y.reshape(-1).astype(np.float32)
    L = len(x)
    cols = 4096
    rows = (L + cols - 1) // cols
    pad = rows * cols - L
    mx = np.concatenate([x, np.full(pad, -10, np.float32)]).reshape(rows, cols)
    my = np.concatenate([y, np.full(pad, -10, np.float32)]).reshape(rows, cols)
    out = np.empty((rows, cols), np.float32)
    for r0 in range(0, rows, 4096):                     # OpenCV map size limit
        out[r0:r0 + 4096] = cv2.remap(img, mx[r0:r0 + 4096], my[r0:r0 + 4096], cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    out = out.reshape(-1)[:L].reshape(shp)
    valid = (x.reshape(shp) >= 0) & (y.reshape(shp) >= 0) & (x.reshape(shp) <= w - 1) & (y.reshape(shp) <= h - 1)
    return out, valid


def refine_mesh(chunk, iterations=3, scale=0.5, max_views=6, search_vox=1.0, steps=9, patch=5, spacing_vox=0.3,
                margin=0.03, smooth=3, chunk_size=100_000):
    """Photo-consistency refinement (Metashape: Refine Mesh, simplified). Every vertex searches along its
    normal for the offset where small surface patches look most alike in its best views (NCC), the offset
    field is smoothed over the mesh, applied, and the search range shrinks each iteration."""
    import time
    from . import raster
    t0 = time.time()
    V, F, C = load_mesh(chunk)
    F = F.astype(np.int64)
    vox = float(chunk.stats.get("mesh", {}).get("voxel") or 0) or mvs.estimate_gsd(chunk, scale) * 3
    cams = chunk.aligned_cameras()
    log("Refine Mesh: %d vertices, %d views, %d iterations" % (len(V), len(cams), iterations))
    # ---- per-view geometry + visibility (z-buffer) -> best views per vertex
    geo, imgs = {}, {}
    Nv = vertex_normals(V, F)
    best_s = np.zeros((len(V), max_views)); best_v = -np.ones((len(V), max_views), int)
    for k, i in enumerate(cams):
        cam = chunk.cameras[i]; sen = chunk.sensors[cam.sensor]
        Ks = cm.K_matrix(sen.intr, scale)
        w, h = int(round(sen.width * scale)), int(round(sen.height * scale))
        Rc = cm.rodrigues_to_R(cam.rvec)[0]; tc = np.asarray(cam.tvec, float)
        uv, z = raster.project_vertices(V, cam.rvec, cam.tvec, Ks)
        ids, zb = raster.rasterize(uv, z, F, w, h)
        ui = np.round(uv[:, 0]).astype(int); vi = np.round(uv[:, 1]).astype(int)
        inb = (z > 0) & (ui >= 1) & (ui < w - 1) & (vi >= 1) & (vi < h - 1)
        vis = np.zeros(len(V), bool)
        vis[inb] = z[inb] <= zb[vi[inb], ui[inb]] * 1.01 + 1e-6
        Cc = -Rc.T @ tc
        d = Cc - V
        dist = np.linalg.norm(d, axis=1)
        cos = (Nv * d).sum(1) / np.maximum(dist, 1e-12)
        sc = np.where(vis & (cos > 0.3), cos * Ks[0, 0] / np.maximum(dist, 1e-12), 0)
        # insert into the per-vertex top-k
        worst = np.argmin(best_s, 1)
        better = sc > best_s[np.arange(len(V)), worst]
        best_s[better, worst[better]] = sc[better]; best_v[better, worst[better]] = k
        geo[k] = (Ks, Rc, tc, w, h)
        if k % 50 == 0:
            log("  visibility %d/%d" % (k + 1, len(cams)))
    o = np.argsort(-best_s, 1)
    best_v = np.take_along_axis(best_v, o, 1); best_s = np.take_along_axis(best_s, o, 1)
    best_v[best_s <= 0] = -1
    used = np.unique(best_v[best_v >= 0])

    def gray(k):
        if k not in imgs:
            i = cams[k]; cam = chunk.cameras[i]; sen = chunk.sensors[cam.sensor]
            img = read_image(chunk.abspath(cam))
            m1, m2, _ = cm.undistort_maps(sen.calib(), sen.width, sen.height, scale)
            g = cv2.cvtColor(cv2.remap(img, m1, m2, cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            imgs[k] = g
        return imgs[k]
    from concurrent.futures import ThreadPoolExecutor
    from .util import n_workers
    with ThreadPoolExecutor(n_workers(8)) as ex:
        list(ex.map(gray, used))
    A = vertex_adjacency(len(V), F).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    g = np.linspace(-1, 1, patch)
    gu, gv = [a.ravel() for a in np.meshgrid(g, g)]
    rng = search_vox * vox
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)
    for it in range(iterations):
        Nv = vertex_normals(V, F)
        t1 = np.cross(Nv, [0, 0, 1.0]); bad = np.linalg.norm(t1, axis=1) < 0.1
        t1[bad] = np.cross(Nv[bad], [1.0, 0, 0])
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True); t2 = np.cross(Nv, t1)
        offs = np.linspace(-rng, rng, steps)
        spacing = spacing_vox * vox
        off_best = np.zeros(len(V)); conf = np.zeros(len(V))
        for a0 in range(0, len(V), chunk_size):
            sl = slice(a0, a0 + chunk_size)
            n = len(V[sl])
            P = (V[sl][:, None, None] + offs[None, :, None, None] * Nv[sl][:, None, None] +
                 spacing * (gu[None, None, :, None] * t1[sl][:, None, None] + gv[None, None, :, None] * t2[sl][:, None, None]))
            S = np.full((n, max_views, steps, patch * patch), np.nan, np.float32)
            bv = best_v[sl]
            for slot in range(max_views):
                for k in np.unique(bv[:, slot]):
                    if k < 0:
                        continue
                    rows = np.where(bv[:, slot] == k)[0]
                    Ks, Rc, tc, w, h = geo[k]
                    Xc = P[rows] @ Rc.T + tc
                    z = np.maximum(Xc[..., 2], 1e-9)
                    x = Xc[..., 0] / z * Ks[0, 0] + Ks[0, 2]; y = Xc[..., 1] / z * Ks[1, 1] + Ks[1, 2]
                    val, ok = _sample_bilinear(gray(k), x, y)
                    val[~ok] = np.nan
                    S[rows, slot] = val
            ref = S[:, 0]                                            # (n, steps, p)
            costs, cnt = np.zeros((n, steps)), np.zeros((n, steps))
            rc = ref - np.nanmean(ref, -1, keepdims=True)
            rn = np.sqrt(np.nansum(rc * rc, -1))
            for slot in range(1, max_views):
                o_ = S[:, slot]
                oc = o_ - np.nanmean(o_, -1, keepdims=True)
                on = np.sqrt(np.nansum(oc * oc, -1))
                ncc = np.nansum(rc * oc, -1) / np.maximum(rn * on, 1e-6)
                ok = np.isfinite(o_).all(-1) & np.isfinite(ref).all(-1) & (rn > 0.02) & (on > 0.02)
                costs += np.where(ok, 1 - ncc, 0); cnt += ok
            good = (cnt >= 1).all(1)
            cost = np.where(cnt > 0, costs / np.maximum(cnt, 1), np.inf)
            b = np.argmin(cost, 1)
            cmin = cost[np.arange(n), b]
            # sub-step parabola
            b0 = np.clip(b, 1, steps - 2)
            c0, c1, c2 = (cost[np.arange(n), b0 - 1], cost[np.arange(n), b0], cost[np.arange(n), b0 + 1])
            den = c0 - 2 * c1 + c2
            fr = np.where(np.isfinite(den) & (den > 1e-6), np.clip(0.5 * (c0 - c2) / np.maximum(den, 1e-6), -0.5, 0.5), 0)
            ob = offs[b0] + fr * (offs[1] - offs[0])
            c_mid = cost[:, steps // 2]
            acc = good & np.isfinite(cmin) & (cmin < 0.6) & (cmin < c_mid - margin)
            off_best[a0:a0 + n] = np.where(acc, ob, 0)
            conf[a0:a0 + n] = np.where(acc, np.clip(1 - cmin, 0, 1), 0)
        # confidence-weighted smoothing of the offset field
        o_ = off_best * conf; w_ = conf.copy()
        for _ in range(smooth):
            o_ = 0.5 * o_ + 0.5 * (A @ o_) / deg
            w_ = 0.5 * w_ + 0.5 * (A @ w_) / deg
        step = np.where(w_ > 1e-6, o_ / np.maximum(w_, 1e-6), 0)
        V = V + step[:, None] * Nv
        log("  iteration %d: %.1f%% of vertices moved, median |offset| %.3g (range %.3g)" %
            (it + 1, 100 * (conf > 0).mean(), np.median(np.abs(step[conf > 0])) if (conf > 0).any() else 0, rng))
        rng *= 0.6
    V = taubin_smooth(V, F, 1)
    import shutil
    src = chunk.path("mesh.ply")
    if not os.path.exists(chunk.path("mesh_unrefined.ply")):
        shutil.copy2(src, chunk.path("mesh_unrefined.ply"))
    write_ply(src, V, faces=F, colors=C)
    chunk.stats["mesh"] = dict(chunk.stats.get("mesh", {}), refined=iterations)
    chunk.stats.pop("texture", None)
    chunk.add_log("Refine Mesh", time.time() - t0, iterations=iterations)
    log("Refine Mesh done in %.0f s" % (time.time() - t0))
    return V
