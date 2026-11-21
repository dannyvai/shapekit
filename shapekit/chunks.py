"""Multi-chunk projects (Metashape: Add / Duplicate / Align Chunks / Merge Chunks).

Every chunk has its own internal frame. chunk.transform (4x4 similarity, internal -> world/common frame) puts
chunks in a common frame: Align Chunks sets it from shared markers or shared photos, reference markers set it
from survey coordinates. Merge Chunks maps everything into the first chunk's internal frame and writes a new
chunk (photos are referenced, not copied).
"""
import os
import json
import shutil
import numpy as np

from . import camera as cm
from .markers import get_transform, umeyama
from .project import Chunk, Sensor, Camera
from .util import log


def _unique_dirname(project):
    used = {c.dirname for c in project.chunks}
    k = len(project.chunks)
    while "chunk_%d" % k in used or os.path.exists(os.path.join(project.dir, "chunk_%d" % k)):
        k += 1
    return "chunk_%d" % k


def add_chunk(project, label=None):
    label = label or "Chunk %d" % (len(project.chunks) + 1)
    ch = Chunk(project, label, _unique_dirname(project))
    os.makedirs(ch.dir, exist_ok=True)
    project.chunks.append(ch)
    return ch


def _relink(cam, src_chunk, dst_dir):
    """Camera path relative to the new chunk folder (photos stay where they are)."""
    p = src_chunk.abspath(cam)
    try:
        return os.path.relpath(p, dst_dir).replace("\\", "/")
    except ValueError:                         # different drive on Windows
        return os.path.abspath(p).replace("\\", "/")


def duplicate_chunk(project, idx, label=None, items=("features", "alignment", "depth", "cloud", "mesh", "texture")):
    """Copy a chunk (Metashape: Duplicate Chunk). Photos are referenced, products are copied."""
    src = project.chunks[idx]
    dst = add_chunk(project, label or src.label + " copy")
    d = json.loads(json.dumps(src.to_dict()))      # deep copy
    tmp = Chunk.from_dict(project, d)
    dst.sensors, dst.markers, dst.scalebars = tmp.sensors, tmp.markers, tmp.scalebars
    dst.region, dst.transform = tmp.region, tmp.transform
    dst.stats = {k: v for k, v in tmp.stats.items()}
    dst.log = list(tmp.log)
    dst.cameras = tmp.cameras
    for c, sc in zip(dst.cameras, src.cameras):
        c.path = _relink(sc, src, dst.dir)
    if "alignment" not in items:
        for c in dst.cameras:
            c.rvec = c.tvec = None
        for k in ("align", "depth_maps", "point_cloud", "mesh", "texture"):
            dst.stats.pop(k, None)
    copy = []
    if "features" in items:
        copy += ["features", "matches.npz"]
    if "alignment" in items:
        copy += ["sparse.npz"]
    if "depth" in items:
        copy += ["depth"]
    if "cloud" in items:
        copy += ["point_cloud.ply"]
    if "mesh" in items:
        copy += ["mesh.ply"]
    if "texture" in items:
        copy += ["model"]
    copy += ["masks"]
    for name in copy:
        s = os.path.join(src.dir, name)
        t = os.path.join(dst.dir, name)
        if os.path.isdir(s):
            shutil.copytree(s, t, dirs_exist_ok=True)
        elif os.path.exists(s):
            shutil.copy2(s, t)
    for k, prod in (("depth", "depth_maps"), ("cloud", "point_cloud"), ("mesh", "mesh"), ("texture", "texture")):
        if k not in items:
            dst.stats.pop(prod, None)
    log("Duplicated '%s' -> '%s' (%s)" % (src.label, dst.label, ", ".join(items)))
    return dst


def remove_chunk(project, idx):
    """Remove a chunk from the project (its folder is kept on disk, renamed *_removed)."""
    ch = project.chunks.pop(idx)
    try:
        os.replace(ch.dir, ch.dir + "_removed")
    except OSError:
        pass
    return ch


# ----------------------------------------------------------------------------- align chunks
def _photo_key(chunk, cam):
    return os.path.normcase(os.path.abspath(chunk.abspath(cam)))


def chunk_correspondences(a, b, method="markers"):
    """Corresponding internal 3D points of chunk a and chunk b: shared marker labels (method='markers') or
    shared photos (camera centres, method='cameras')."""
    P, Q, names = [], [], []
    if method == "markers":
        mb = {m["label"]: m for m in b.markers or [] if m.get("position") is not None}
        for m in a.markers or []:
            o = mb.get(m["label"])
            if m.get("position") is not None and o is not None:
                P.append(m["position"]); Q.append(o["position"]); names.append(m["label"])
    elif method == "cameras":
        kb = {_photo_key(b, c): c for c in b.cameras if c.aligned}
        for c in a.cameras:
            o = kb.get(_photo_key(a, c))
            if c.aligned and o is not None:
                P.append(c.center); Q.append(o.center); names.append(c.label)
    else:
        raise ValueError(method)
    return np.array(P, float).reshape(-1, 3), np.array(Q, float).reshape(-1, 3), names


def align_chunks(project, ref_idx, idxs, method="markers"):
    """Set the transforms of chunks idxs so they share the frame of chunk ref_idx (Metashape: Align Chunks).
    Returns {idx: dict(n, rmse)} -- rmse in the reference chunk's world units."""
    ref = project.chunks[ref_idx]
    Tref = get_transform(ref)
    out = {}
    for i in idxs:
        if i == ref_idx:
            continue
        ch = project.chunks[i]
        P, Q, names = chunk_correspondences(ref, ch, method)
        if len(P) < 3:
            raise RuntimeError("'%s' and '%s' share only %d %s (need >= 3)" %
                               (ref.label, ch.label, len(P), "markers" if method == "markers" else "aligned photos"))
        Pw = P @ Tref[:3, :3].T + Tref[:3, 3]
        # robust: drop the worst correspondence while residuals are far above the median
        keep = np.ones(len(P), bool)
        for _ in range(max(0, len(P) - 3)):
            s, R, t = umeyama(Q[keep], Pw[keep])
            res = np.linalg.norm((Q @ R.T) * s + t - Pw, axis=1)
            med = np.median(res[keep]) + 1e-12
            worst = np.argmax(np.where(keep, res, -1))
            if res[worst] > 5 * med and keep.sum() > 3:
                keep[worst] = False
            else:
                break
        s, R, t = umeyama(Q[keep], Pw[keep])
        T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
        res = np.linalg.norm(Q[keep] @ T[:3, :3].T + T[:3, 3] - Pw[keep], axis=1)
        ch.transform = T.tolist()
        ch.stats["reference"] = dict(mode="aligned to chunk '%s' by %s (%d)" % (ref.label, method, keep.sum()),
                                     scale=float(np.cbrt(abs(np.linalg.det(T[:3, :3])))),
                                     marker_rmse=float(np.sqrt((res ** 2).mean())), scalebar_rmse=None)
        out[i] = dict(n=int(keep.sum()), rmse=float(np.sqrt((res ** 2).mean())), dropped=[n for n, k in zip(names, keep) if not k])
        log("Align Chunks: '%s' -> '%s' by %d %s, RMSE %.4g" % (ch.label, ref.label, keep.sum(), method, out[i]["rmse"]))
    return out


# ----------------------------------------------------------------------------- merge chunks
def _apply_pose(rvec, tvec, S):
    """Camera pose under X_new = s R X_old + t: R' = Rc R^T, t' = s tc - R' t (camera units scale by s)."""
    s = float(np.cbrt(abs(np.linalg.det(S[:3, :3]))))
    R = S[:3, :3] / s
    Rc = cm.rodrigues_to_R(rvec)[0]
    Rn = Rc @ R.T
    tn = s * np.asarray(tvec, float) - Rn @ S[:3, 3]
    return cm.R_to_rodrigues(Rn), tn, s


def merge_chunks(project, idxs, label="Merged Chunk", merge_markers=True, merge_depth=True, merge_cloud=True,
                 merge_mesh=False):
    """Merge aligned chunks into a new chunk in the first chunk's frame (Metashape: Merge Chunks).
    Depth maps are copied (rescaled) so the merged chunk can build a single point cloud / mesh."""
    from .io import read_ply, write_ply
    chunks = [project.chunks[i] for i in idxs]
    base = chunks[0]
    Tb_inv = np.linalg.inv(get_transform(base))
    out = add_chunk(project, label)
    out.transform = base.transform
    os.makedirs(os.path.join(out.dir, "depth"), exist_ok=True)
    sp_parts, mk = [], {}
    cloud = []
    meshes = []
    for ch in chunks:
        S = Tb_inv @ get_transform(ch)            # chunk internal -> base internal
        s = float(np.cbrt(abs(np.linalg.det(S[:3, :3]))))
        soff, coff = len(out.sensors), len(out.cameras)
        for sen in ch.sensors:
            out.sensors.append(Sensor(sen.label, sen.width, sen.height, sen.intr.copy(), sen.fixed, sen.model))
        for i, c in enumerate(ch.cameras):
            nc = Camera(c.label, _relink(c, ch, out.dir), c.sensor + soff, enabled=c.enabled, quality=c.quality,
                        source=c.source, frame=c.frame)
            if c.aligned:
                nc.rvec, nc.tvec, _ = _apply_pose(c.rvec, c.tvec, S)
                if merge_depth:
                    for suf in ("", "_f"):
                        p = os.path.join(ch.dir, "depth", "%05d%s.npz" % (i, suf))
                        if os.path.exists(p):
                            d = dict(np.load(p))
                            if suf == "":
                                d["depth"] = (d["depth"].astype(np.float32) * s).astype(np.float16)
                            np.savez(os.path.join(out.dir, "depth", "%05d%s.npz" % (i + coff, suf)), **d)
            mp = os.path.join(ch.dir, "masks", "%05d.png" % i)
            if os.path.exists(mp):
                os.makedirs(os.path.join(out.dir, "masks"), exist_ok=True)
                shutil.copy2(mp, os.path.join(out.dir, "masks", "%05d.png" % (i + coff)))
            out.cameras.append(nc)
        sp = ch.load_sparse()
        if sp is not None:
            sp_parts.append((sp, S, coff))
        if merge_markers:
            for m in ch.markers or []:
                pos = None if m.get("position") is None else (np.asarray(m["position"]) @ S[:3, :3].T + S[:3, 3])
                proj = {str(int(k) + coff): v for k, v in m.get("projections", {}).items()}
                if m["label"] in mk:
                    o = mk[m["label"]]
                    o["projections"].update(proj)
                    if pos is not None:
                        o["_pos"].append(pos)
                else:
                    n = json.loads(json.dumps(m))
                    n["projections"] = proj
                    n["_pos"] = [pos] if pos is not None else []
                    mk[m["label"]] = n
        if merge_cloud and os.path.exists(os.path.join(ch.dir, "point_cloud.ply")):
            from .io import ply_vertex_count
            if ply_vertex_count(os.path.join(ch.dir, "point_cloud.ply")) > 30_000_000:
                log("  point cloud of '%s' is very large - not merged; build the point cloud of the merged chunk "
                    "from its merged depth maps instead" % ch.label)
                merge_cloud = False
                cloud = []
                continue_cloud = False
            else:
                continue_cloud = True
        if merge_cloud and os.path.exists(os.path.join(ch.dir, "point_cloud.ply")) and continue_cloud:
            pc = read_ply(os.path.join(ch.dir, "point_cloud.ply"))
            P = pc["verts"] @ S[:3, :3].T + S[:3, 3]
            N = pc["normals"] @ (S[:3, :3] / s).T if "normals" in pc else None
            cloud.append((P, pc.get("colors"), N, pc.get("confidence")))
        if merge_mesh and os.path.exists(os.path.join(ch.dir, "mesh.ply")):
            m_ = read_ply(os.path.join(ch.dir, "mesh.ply"))
            meshes.append((m_["verts"] @ S[:3, :3].T + S[:3, 3], m_["faces"], m_.get("colors")))
    # sparse cloud
    if sp_parts:
        pts, col, oc, op, uv, err = [], [], [], [], [], []
        poff = 0
        for sp, S, coff in sp_parts:
            pts.append(sp["points"] @ S[:3, :3].T + S[:3, 3]); col.append(sp["colors"])
            oc.append(sp["obs_cam"] + coff); op.append(sp["obs_pt"] + poff); uv.append(sp["obs_uv"])
            err.append(sp["obs_err"]); poff += len(sp["points"])
        out.save_sparse(points=np.concatenate(pts), colors=np.concatenate(col), obs_cam=np.concatenate(oc),
                        obs_pt=np.concatenate(op), obs_uv=np.concatenate(uv), obs_err=np.concatenate(err))
        e = np.concatenate(err)
        out.stats["align"] = dict(cameras=len(out.cameras), aligned=len(out.aligned_cameras()),
                                  tie_points=int(poff), projections=int(len(e)),
                                  rms_reprojection_px=float(np.sqrt((e ** 2).mean())) if len(e) else 0.0)
    for m in mk.values():
        ps = m.pop("_pos")
        m["position"] = np.mean(ps, 0).tolist() if ps else None
    out.markers = list(mk.values())
    out.scalebars = [dict(b) for ch in chunks for b in ch.scalebars or []]
    if merge_depth and any(os.listdir(os.path.join(out.dir, "depth"))):
        out.stats["depth_maps"] = dict(base.stats.get("depth_maps", {}), cameras=len(out.aligned_cameras()))
    if cloud:
        P = np.concatenate([c[0] for c in cloud])
        C = np.concatenate([c[1] for c in cloud]) if all(c[1] is not None for c in cloud) else None
        N = np.concatenate([c[2] for c in cloud]) if all(c[2] is not None for c in cloud) else None
        Q = np.concatenate([c[3] for c in cloud]) if all(c[3] is not None for c in cloud) else None
        write_ply(out.path("point_cloud.ply"), P, colors=C, normals=N,
                  extra={"confidence": Q.astype(np.uint8)} if Q is not None else None)
        out.stats["point_cloud"] = dict(points=int(len(P)), spacing=base.stats.get("point_cloud", {}).get("spacing"))
    if meshes:
        V, F, C, off = [], [], [], 0
        for v, f, c in meshes:
            V.append(v); F.append(f + off); C.append(c if c is not None else np.full((len(v), 3), 128, np.uint8)); off += len(v)
        write_ply(out.path("mesh.ply"), np.concatenate(V), faces=np.concatenate(F), colors=np.concatenate(C))
        out.stats["mesh"] = dict(vertices=int(off), faces=int(sum(len(f) for f in F)))
    from .sfm import region_from_sparse
    if out.load_sparse() is not None:
        region_from_sparse(out)
    log("Merged %d chunks into '%s': %d cameras (%d aligned)" % (len(chunks), out.label, len(out.cameras),
                                                                 len(out.aligned_cameras())))
    return out
