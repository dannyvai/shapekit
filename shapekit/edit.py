"""GUI-independent editing operations: screen-space selection, deleting / cropping tie points, dense
points and mesh faces (keeping the texture valid), gradual selection and region maths."""
import os
import numpy as np

from .io import read_ply, write_ply
from .util import log


# ----------------------------------------------------------------------------- screen-space selection
def project_to_screen(X, M, width, height):
    """X (N,3) world, M 4x4 composite projection (VTK convention), -> (N,2) screen px (top-left origin), w>0 mask."""
    Xh = np.c_[X, np.ones(len(X))] @ np.asarray(M).T
    w = Xh[:, 3]
    ok = w > 1e-9
    ndc = Xh[:, :2] / np.where(ok, w, 1)[:, None]
    sx = (ndc[:, 0] + 1) * 0.5 * width
    sy = (1 - (ndc[:, 1] + 1) * 0.5) * height
    return np.stack([sx, sy], 1), ok & (np.abs(Xh[:, 2] / np.where(ok, w, 1)) <= 1)


def points_in_polygon(P, poly):
    """Vectorised even-odd rule. P (N,2), poly (K,2)."""
    poly = np.asarray(poly, float)
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), bool)
    xb0, yb0, xb1, yb1 = poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()
    cand = (x >= xb0) & (x <= xb1) & (y >= yb0) & (y <= yb1)
    idx = np.where(cand)[0]
    xs, ys = x[idx], y[idx]
    res = np.zeros(len(idx), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > ys) != (yj > ys)) & (xs < (xj - xi) * (ys - yi) / ((yj - yi) if yj != yi else 1e-12) + xi)
        res ^= cond
        j = i
    inside[idx] = res
    return inside


def select_in_polygon(X, M, width, height, poly):
    P, ok = project_to_screen(X, M, width, height)
    return points_in_polygon(P, poly) & ok


def combine(old, new, mode):
    """mode: 'replace' | 'add' | 'subtract'"""
    if old is None or mode == "replace" or len(old) != len(new):
        return new
    return (old | new) if mode == "add" else (old & ~new)


# ----------------------------------------------------------------------------- tie points
def delete_tie_points(chunk, mask):
    sp = chunk.load_sparse()
    keep = ~np.asarray(mask, bool)
    remap = -np.ones(len(keep), int); remap[keep] = np.arange(keep.sum())
    ok = keep[sp["obs_pt"]]
    out = dict(points=sp["points"][keep], colors=sp["colors"][keep])
    for k in ("obs_cam", "obs_uv", "obs_err"):
        out[k] = sp[k][ok]
    out["obs_pt"] = remap[sp["obs_pt"][ok]]
    chunk.save_sparse(**out)
    chunk.stats.setdefault("align", {})["tie_points"] = int(keep.sum())
    log("Deleted %d tie points (%d left)" % ((~keep).sum(), keep.sum()))


def tie_point_errors(chunk, criterion="reprojection"):
    """Per-point value for Gradual Selection: max reprojection error (px) or image count."""
    sp = chunk.load_sparse()
    n = len(sp["points"])
    if criterion == "image_count":
        return np.bincount(sp["obs_pt"], minlength=n).astype(float)
    e = np.zeros(n)
    np.maximum.at(e, sp["obs_pt"], sp["obs_err"].astype(float))
    return e


# ----------------------------------------------------------------------------- dense cloud
def delete_dense_points(chunk, mask):
    p = chunk.path("point_cloud.ply")
    d = read_ply(p)
    keep = ~np.asarray(mask, bool)
    write_ply(p, d["verts"][keep], colors=d.get("colors")[keep] if d.get("colors") is not None else None,
              normals=d["normals"][keep] if d.get("normals") is not None else None,
              extra={"confidence": d["confidence"][keep]} if "confidence" in d else None)
    chunk.stats.setdefault("point_cloud", {})["points"] = int(keep.sum())
    log("Deleted %d dense points (%d left)" % ((~keep).sum(), keep.sum()))


def filter_dense_by_confidence(chunk, min_conf):
    d = read_ply(chunk.path("point_cloud.ply"))
    if "confidence" not in d:
        return 0
    m = d["confidence"] < min_conf
    delete_dense_points(chunk, m)
    return int(m.sum())


# ----------------------------------------------------------------------------- mesh
def delete_faces(chunk, face_mask):
    """Remove faces; drop unused vertices; keep texture UVs consistent (uv.npz face_uv is per face)."""
    p = chunk.path("mesh.ply")
    d = read_ply(p)
    V, F, C = d["verts"], d["faces"], d.get("colors")
    keep = ~np.asarray(face_mask, bool)
    F2 = F[keep]
    used = np.unique(F2)
    remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    write_ply(p, V[used], faces=remap[F2], colors=C[used] if C is not None else None)
    uvp = chunk.path("model", "uv.npz")
    if os.path.exists(uvp):
        u = np.load(uvp)
        if len(u["face_uv"]) == len(F):
            np.savez_compressed(uvp, uv=u["uv"], face_uv=u["face_uv"][keep])
        else:
            os.remove(uvp)        # stale: texture must be rebuilt
    chunk.stats.setdefault("mesh", {}).update(faces=int(keep.sum()), vertices=int(len(used)))
    log("Deleted %d faces (%d left)" % ((~keep).sum(), keep.sum()))


def faces_from_vertex_mask(F, vmask, mode="any"):
    sel = vmask[F]
    return sel.any(1) if mode == "any" else sel.all(1)


def small_component_faces(chunk, frac):
    from .mesh import remove_small_components, vertex_adjacency
    from scipy.sparse.csgraph import connected_components
    d = read_ply(chunk.path("mesh.ply"))
    A = vertex_adjacency(len(d["verts"]), d["faces"])
    n, lab = connected_components(A, directed=False)
    fl = lab[d["faces"][:, 0]]
    cnt = np.bincount(fl, minlength=n)
    return cnt[fl] < frac * cnt.max()


# ----------------------------------------------------------------------------- region
def region_matrix(region):
    """4x4 mapping the unit cube [-0.5, 0.5]^3 onto the region box."""
    c = np.array(region["center"]); s = np.array(region["size"]); R = np.array(region["R"])
    T = np.eye(4)
    T[:3, :3] = R.T @ np.diag(s)
    T[:3, 3] = c
    return T


def region_from_matrix(T):
    T = np.asarray(T, float)
    A = T[:3, :3]
    size = np.linalg.norm(A, axis=0)
    axes = A / np.maximum(size, 1e-12)
    # orthonormalise (widget transforms can accumulate tiny shear)
    U, _, Vt = np.linalg.svd(axes)
    axes = U @ Vt
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    return dict(center=T[:3, 3].tolist(), size=size.tolist(), R=axes.T.tolist())


def region_corners(region):
    T = region_matrix(region)
    cube = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    return cube @ T[:3, :3].T + T[:3, 3]
