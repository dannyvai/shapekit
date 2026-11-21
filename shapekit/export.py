"""Export Model / Export Points / Export Cameras."""
import os
import numpy as np
import cv2

from .io import write_obj, write_glb, write_ply, read_ply, export_colmap
from .mesh import load_mesh, decimate_clustering
from .util import log


def viewer_transform(chunk):
    """Internal -> viewer frame: region centre at origin, 'up' -> +Y (glTF), real-world units if scaled.
    Use as (X - c) @ M.T"""
    from .markers import get_transform
    c = np.array(chunk.region["center"])
    to_y_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    if getattr(chunk, "transform", None) is not None:
        T = get_transform(chunk)
        return c, to_y_up @ T[:3, :3]          # world Z up (scaled / georeferenced)
    return c, to_y_up @ np.array(chunk.region["R"])


def world(chunk, X):
    """Internal coordinates -> chunk world coordinates (identity if no transform)."""
    from .markers import get_transform, apply_transform
    return apply_transform(get_transform(chunk), X) if getattr(chunk, "transform", None) is not None else np.asarray(X)


def world_normals(chunk, N):
    from .markers import get_transform
    if getattr(chunk, "transform", None) is None or N is None:
        return N
    R = get_transform(chunk)[:3, :3]
    N = np.asarray(N) @ R.T
    return (N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def export_model(chunk, path, max_faces=None, texture_size=None, local_frame=False):
    """.obj (textured if a texture exists), .glb (textured / vertex colours), .ply (vertex colours)."""
    V, F, C = load_mesh(chunk)
    uvp = chunk.path("model", "uv.npz")
    textured = os.path.exists(uvp)
    if local_frame:
        c, M = viewer_transform(chunk)
        V = (V - c) @ M.T
    else:
        V = world(chunk, V)
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if max_faces and len(F) > max_faces:
        V, F, C = decimate_clustering(V, F, max_faces, C)
        textured = False
        log("  decimated to %d faces (vertex colours only)" % len(F))
    if ext == ".ply":
        write_ply(path, V, faces=F, colors=None if C is None else np.asarray(C).clip(0, 255).astype(np.uint8))
    elif ext == ".obj":
        if textured:
            d = np.load(uvp)
            tex_src = chunk.path("model", "model_texture.jpg")
            tex_dst = os.path.splitext(path)[0] + "_texture.jpg"
            img = cv2.imread(tex_src)
            if texture_size and texture_size < img.shape[0]:
                img = cv2.resize(img, (texture_size, texture_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(tex_dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            write_obj(path, V, F, d["uv"], d["face_uv"], texture_file=tex_dst)
        else:
            write_obj(path, V, F, colors=C)
    elif ext == ".glb":
        if textured:
            d = np.load(uvp)
            img = cv2.imread(chunk.path("model", "model_texture.jpg"))
            if texture_size and texture_size < img.shape[0]:
                img = cv2.resize(img, (texture_size, texture_size), interpolation=cv2.INTER_AREA)
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            write_glb(path, V, F, d["uv"], d["face_uv"], texture_jpg=jpg)
        else:
            write_glb(path, V, F, colors=C)
    else:
        raise ValueError("unsupported model format %s (use .obj .glb .ply)" % ext)
    log("Exported model -> %s" % path)
    return path


def export_points(chunk, path, source="dense"):
    src = chunk.path("point_cloud.ply") if source == "dense" else None
    if source == "dense" and os.path.exists(src):
        d = read_ply(src)
        write_ply(path, world(chunk, d["verts"]), colors=d.get("colors"), normals=world_normals(chunk, d.get("normals")),
                  extra={"confidence": d["confidence"]} if "confidence" in d else None)
    else:
        sp = chunk.load_sparse()
        write_ply(path, world(chunk, sp["points"]), colors=sp["colors"])
    log("Exported points -> %s" % path)


def export_cameras(chunk, out_dir, fmt="colmap", copy_images=True):
    if fmt != "colmap":
        raise ValueError("only colmap supported")
    sparse = os.path.join(out_dir, "sparse", "0")
    export_colmap(chunk, sparse)
    if copy_images:
        import shutil
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        for i in chunk.aligned_cameras():
            c = chunk.cameras[i]
            dst = os.path.join(img_dir, os.path.basename(c.path))
            if not os.path.exists(dst):
                shutil.copy2(chunk.abspath(c), dst)
    log("Exported cameras (COLMAP) -> %s" % out_dir)


def export_markers(chunk, path):
    """Markers as CSV: label, code, estimated world XYZ, reference XYZ, error, #projections."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "code", "x", "y", "z", "ref_x", "ref_y", "ref_z", "error", "projections", "reproj_px"])
        for m in chunk.markers:
            if m.get("position") is None:
                continue
            X = world(chunk, np.array(m["position"])[None])[0]
            ref = (m.get("reference") or {}).get("xyz") or ["", "", ""]
            err = float(np.linalg.norm(X - np.array(ref))) if m.get("reference") else ""
            w.writerow([m["label"], m.get("code") or "", *np.round(X, 5), *ref, err, len(m["projections"]),
                        round(m.get("error_px") or 0, 3)])
    log("Exported markers -> %s" % path)
