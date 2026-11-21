"""Readers/writers: PLY (points / mesh), OBJ+MTL, binary glTF (GLB), COLMAP text model."""
import json
import os
import struct
import numpy as np
import cv2

from . import camera as cm


def write_ply(path, verts, faces=None, colors=None, normals=None, extra=None):
    verts = np.asarray(verts, np.float32)
    n = len(verts)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    extra = extra or {}
    for k, v in extra.items():
        fields.append((k, np.asarray(v).dtype.str.replace("|", "")))
    arr = np.zeros(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = verts.T
    if normals is not None:
        arr["nx"], arr["ny"], arr["nz"] = np.asarray(normals, np.float32).T
    if colors is not None:
        c = np.asarray(colors, np.uint8)
        arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
    for k, v in extra.items():
        arr[k] = v
    tmap = {"<f4": "float", "u1": "uchar", "<i4": "int", "<u4": "uint", "<f8": "double", "<u2": "ushort"}
    hdr = ["ply", "format binary_little_endian 1.0", "element vertex %d" % n]
    hdr += ["property %s %s" % (tmap[t], name) for name, t in fields]
    if faces is not None:
        hdr += ["element face %d" % len(faces), "property list uchar int vertex_indices"]
    hdr.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode())
        f.write(arr.tobytes())
        if faces is not None:
            fa = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "<i4", 3)])
            fa["n"] = 3; fa["i"] = faces
            f.write(fa.tobytes())


def ply_header(path):
    """(vertex dtype, n_vertices, n_faces, header byte length) of a binary PLY."""
    with open(path, "rb") as f:
        hdr = []
        while True:
            line = f.readline().decode().strip()
            hdr.append(line)
            if line == "end_header":
                break
        off = f.tell()
    tmap = {"float": "<f4", "uchar": "u1", "int": "<i4", "uint": "<u4", "double": "<f8", "ushort": "<u2"}
    nv = nf = 0; fields = []; cur = None
    for l in hdr:
        p = l.split()
        if p[:2] == ["element", "vertex"]:
            nv = int(p[2]); cur = "v"
        elif p[:2] == ["element", "face"]:
            nf = int(p[2]); cur = "f"
        elif p and p[0] == "property" and cur == "v":
            fields.append((p[2], tmap[p[1]]))
    return np.dtype(fields), nv, nf, off


def ply_vertex_count(path):
    return ply_header(path)[1]


def read_ply_sampled(path, max_points):
    """Every k-th vertex of a (possibly huge) point-cloud PLY via a memory map, at most ~max_points.
    Returns the same dict as read_ply plus 'stride'."""
    dt, nv, nf, off = ply_header(path)
    if nv <= max_points or nf:
        d = read_ply(path); d["stride"] = 1
        return d
    k = int(np.ceil(nv / max_points))
    v = np.array(np.memmap(path, dtype=dt, mode="r", offset=off, shape=(nv,))[::k])
    out = dict(verts=np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), faces=None, stride=k)
    if "red" in v.dtype.names:
        out["colors"] = np.stack([v["red"], v["green"], v["blue"]], 1)
    if "nx" in v.dtype.names:
        out["normals"] = np.stack([v["nx"], v["ny"], v["nz"]], 1)
    for name in v.dtype.names:
        if name not in ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"):
            out[name] = np.array(v[name])
    return out


def transform_ply_stream(src, dst, fn_points, fn_normals=None, block=4_000_000):
    """Copy a vertex-only PLY applying fn_points (N,3)->(N,3) (and fn_normals) block by block (constant memory)."""
    dt, nv, nf, off = ply_header(src)
    if nf:
        raise ValueError("transform_ply_stream: vertex-only PLY expected")
    with open(src, "rb") as f:
        header = f.read(off)
    mm = np.memmap(src, dtype=dt, mode="r", offset=off, shape=(nv,))
    with open(dst, "wb") as f:
        f.write(header)
        for a in range(0, nv, block):
            v = np.array(mm[a:a + block])
            P = fn_points(np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)).astype(np.float32)
            v["x"], v["y"], v["z"] = P.T
            if fn_normals is not None and "nx" in v.dtype.names:
                N = fn_normals(np.stack([v["nx"], v["ny"], v["nz"]], 1)).astype(np.float32)
                v["nx"], v["ny"], v["nz"] = N.T
            f.write(v.tobytes())


def read_ply(path):
    with open(path, "rb") as f:
        hdr = []
        while True:
            line = f.readline().decode().strip()
            hdr.append(line)
            if line == "end_header":
                break
        tmap = {"float": "<f4", "uchar": "u1", "int": "<i4", "uint": "<u4", "double": "<f8", "ushort": "<u2"}
        nv = nf = 0; fields = []; cur = None
        for l in hdr:
            p = l.split()
            if p[:2] == ["element", "vertex"]:
                nv = int(p[2]); cur = "v"
            elif p[:2] == ["element", "face"]:
                nf = int(p[2]); cur = "f"
            elif p[0] == "property" and cur == "v":
                fields.append((p[2], tmap[p[1]]))
        v = np.frombuffer(f.read(nv * np.dtype(fields).itemsize), dtype=fields)
        faces = None
        if nf:
            fa = np.frombuffer(f.read(nf * 13), dtype=[("n", "u1"), ("i", "<i4", 3)])
            faces = fa["i"].copy()
    out = dict(verts=np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), faces=faces)
    if "red" in v.dtype.names:
        out["colors"] = np.stack([v["red"], v["green"], v["blue"]], 1)
    if "nx" in v.dtype.names:
        out["normals"] = np.stack([v["nx"], v["ny"], v["nz"]], 1)
    for k in v.dtype.names:
        if k not in ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"):
            out[k] = np.array(v[k])
    return out


def write_obj(path, verts, faces, uvs=None, face_uv=None, texture_file=None, colors=None):
    base = os.path.splitext(path)[0]
    mtl = os.path.basename(base) + ".mtl"
    with open(path, "w") as f:
        f.write("# shapekit\n")
        if texture_file:
            f.write("mtllib %s\n" % mtl)
        if colors is not None:
            vc = np.hstack([verts, colors / 255.0])
            np.savetxt(f, vc, fmt="v %.6f %.6f %.6f %.4f %.4f %.4f")
        else:
            np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
        if uvs is not None:
            np.savetxt(f, uvs, fmt="vt %.6f %.6f")
            f.write("usemtl material0\n")
            fi = np.hstack([faces[:, :1] + 1, face_uv[:, :1] + 1, faces[:, 1:2] + 1, face_uv[:, 1:2] + 1,
                            faces[:, 2:3] + 1, face_uv[:, 2:3] + 1])
            np.savetxt(f, fi, fmt="f %d/%d %d/%d %d/%d")
        else:
            np.savetxt(f, faces + 1, fmt="f %d %d %d")
    if texture_file:
        with open(base + ".mtl", "w") as f:
            f.write("newmtl material0\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd %s\n" % os.path.basename(texture_file))


def write_glb(path, verts, faces, uvs=None, face_uv=None, texture_jpg=None, colors=None, y_up_R=None):
    """Binary glTF. If textured, vertices are split per (vertex, uv) pair."""
    if uvs is not None:
        cu = uvs[face_uv.ravel()]
        q = np.round(cu * 2 ** 20).astype(np.int64)
        pair = np.stack([faces.ravel().astype(np.int64), q[:, 0], q[:, 1]], 1)
        uniq, first, inv = np.unique(pair, axis=0, return_index=True, return_inverse=True)
        pos = verts[uniq[:, 0]].astype(np.float32)
        tex = cu[first].astype(np.float32).copy()
        tex[:, 1] = 1 - tex[:, 1]
        idx = inv.reshape(-1).astype(np.uint32)
        col = None
    else:
        pos, tex, idx = verts.astype(np.float32), None, faces.ravel().astype(np.uint32)
        col = None if colors is None else (np.asarray(colors)[:, :3]).astype(np.uint8)
    if y_up_R is not None:
        pos = pos @ np.asarray(y_up_R).T
    bufs, views, accs = [], [], []
    off = 0

    def add(data, target, comp, typ, count, minmax=False):
        nonlocal off
        b = data.tobytes()
        pad = (-len(b)) % 4
        views.append(dict(buffer=0, byteOffset=off, byteLength=len(b), **({"target": target} if target else {})))
        acc = dict(bufferView=len(views) - 1, componentType=comp, count=count, type=typ)
        if minmax:
            acc["min"] = data.min(0).tolist(); acc["max"] = data.max(0).tolist()
        if comp == 5121 and typ == "VEC3":
            acc["normalized"] = True
        accs.append(acc)
        bufs.append(b + b"\0" * pad); off += len(b) + pad
        return len(accs) - 1

    attrs = {"POSITION": add(pos, 34962, 5126, "VEC3", len(pos), True)}
    if tex is not None:
        attrs["TEXCOORD_0"] = add(tex, 34962, 5126, "VEC2", len(tex))
    if col is not None:
        c4 = np.hstack([col, np.full((len(col), 1), 255, np.uint8)])
        attrs["COLOR_0"] = add(c4, 34962, 5121, "VEC4", len(c4))
        accs[-1]["normalized"] = True
    ind = add(idx, 34963, 5125, "SCALAR", len(idx))
    gl = dict(asset=dict(version="2.0", generator="shapekit"), scene=0, scenes=[dict(nodes=[0])],
              nodes=[dict(mesh=0)], accessors=accs, bufferViews=views)
    prim = dict(attributes=attrs, indices=ind)
    if texture_jpg is not None:
        b = texture_jpg
        pad = (-len(b)) % 4
        views.append(dict(buffer=0, byteOffset=off, byteLength=len(b)))
        bufs.append(b + b"\0" * pad); off += len(b) + pad
        gl["images"] = [dict(bufferView=len(views) - 1, mimeType="image/jpeg")]
        gl["samplers"] = [dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)]
        gl["textures"] = [dict(source=0, sampler=0)]
        gl["materials"] = [dict(pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0,
                                                          roughnessFactor=1), doubleSided=True)]
        gl["extensionsUsed"] = ["KHR_materials_unlit"]
        gl["materials"][0]["extensions"] = {"KHR_materials_unlit": {}}
        prim["material"] = 0
    else:
        gl["materials"] = [dict(pbrMetallicRoughness=dict(metallicFactor=0, roughnessFactor=1), doubleSided=True,
                                extensions={"KHR_materials_unlit": {}})]
        gl["extensionsUsed"] = ["KHR_materials_unlit"]
        prim["material"] = 0
    gl["meshes"] = [dict(primitives=[prim])]
    binb = b"".join(bufs)
    gl["buffers"] = [dict(byteLength=len(binb))]
    js = json.dumps(gl).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(binb)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
        f.write(struct.pack("<II", len(binb), 0x004E4942)); f.write(binb)


def export_colmap(chunk, out_dir):
    """COLMAP text model (OPENCV camera model) -- usable for Gaussian Splatting / NeRF tools."""
    os.makedirs(out_dir, exist_ok=True)
    sp = chunk.load_sparse()
    from .markers import get_transform
    T = get_transform(chunk) if getattr(chunk, "transform", None) is not None else np.eye(4)
    s_ = float(np.cbrt(np.linalg.det(T[:3, :3]))); Rw = T[:3, :3] / s_; tw = T[:3, 3]
    pts_w = sp["points"] @ T[:3, :3].T + tw
    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        for s, sen in enumerate(chunk.sensors):
            fx, cx, cy, k1, k2, k3, p1, p2 = sen.intr[:8]
            if getattr(sen, "model", "frame") != "frame":
                # equidistant + radial Brown == COLMAP OPENCV_FISHEYE (theta_d = theta(1 + k1 th^2 + ...)); the
                # equisolid model has no COLMAP equivalent and is written the same way (approximation)
                f.write("%d OPENCV_FISHEYE %d %d %.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n" %
                        (s + 1, sen.width, sen.height, fx, fx, cx + 0.5, cy + 0.5, k1, k2, k3, 0.0))
                continue
            f.write("%d OPENCV %d %d %.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n" %
                    (s + 1, sen.width, sen.height, fx, fx, cx + 0.5, cy + 0.5, k1, k2, p1, p2))
    obs_by_cam = {}
    for k, (c, p, uv) in enumerate(zip(sp["obs_cam"], sp["obs_pt"], sp["obs_uv"])):
        obs_by_cam.setdefault(int(c), []).append((uv, int(p)))
    track = {}
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in chunk.aligned_cameras():
            c = chunk.cameras[i]
            R = cm.rodrigues_to_R(c.rvec)[0]
            # world pose: x_c' = s x_c = (R Rw^T) X_w + (s t - R Rw^T tw)
            Rn = R @ Rw.T
            tn = s_ * np.asarray(c.tvec) - Rn @ tw
            q = rot_to_quat(Rn)
            f.write("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %d %s\n" % (i + 1, *q, *tn, c.sensor + 1,
                                                                       os.path.basename(c.path)))
            line = []
            for j, (uv, p) in enumerate(obs_by_cam.get(i, [])):
                line.append("%.2f %.2f %d" % (uv[0] + 0.5, uv[1] + 0.5, p + 1))
                track.setdefault(p, []).append((i + 1, j))
            f.write(" ".join(line) + "\n")
    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for p, (X, col) in enumerate(zip(pts_w, sp["colors"])):
            tr = " ".join("%d %d" % t for t in track.get(p, []))
            f.write("%d %.6f %.6f %.6f %d %d %d 0.5 %s\n" % (p + 1, *X, *col, tr))


def rot_to_quat(R):
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2]); y = np.copysign(y, R[0, 2] - R[2, 0]); z = np.copysign(z, R[1, 0] - R[0, 1])
    return w, x, y, z
