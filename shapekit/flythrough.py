"""Fly-through videos: render the model along a virtual "drone" path and save an MP4 (+ a poster JPG).

Patterns (all in the chunk's frame, "up" = the region's up axis, which faces the cameras):
  orbit       full 360 degree circle around the site at a fixed elevation
  helix       1.5 turns, descending from a high to a low elevation
  flyover     straight pass over the site along its long side, camera pitched down
  reveal      starts low and close on the centre, pulls back and up while turning 90 degrees
  topdown     bird's-eye view slowly rotating and descending
  capture     follows the (smoothed) path of the original photos / video

Rendering is off-screen with VTK (GPU), frames are piped into FFmpeg (H.264, yuv420p, faststart).
"""
import os
import subprocess
import sys
import time
import numpy as np

from .util import log

PATTERNS = ["orbit", "helix", "flyover", "reveal", "topdown", "capture"]
SIZES = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160)}


def _smooth(t):
    return t * t * (3 - 2 * t)


def site_frame(chunk):
    """Centre, axes (R rows: x, y, up) and horizontal half-extent of the region (or the tie points)."""
    if chunk.region:
        c = np.array(chunk.region["center"], float)
        R = np.array(chunk.region["R"], float)
        s = np.array(chunk.region["size"], float)
    else:
        from .sfm import region_from_sparse
        reg = region_from_sparse(chunk)
        c, R, s = np.array(reg["center"]), np.array(reg["R"]), np.array(reg["size"])
    return c, R, s


def camera_path(chunk, pattern="orbit", n_frames=600, elevation=30.0, distance=1.0, start_angle="auto", fov=50.0,
                arc=360.0):
    """Returns eye (n,3), target (n,3), up (n,3) for every frame.
    start_angle 'auto' = the side the photos were taken from (sites scanned from one side look best from there).
    arc: orbit sweep in degrees; < 360 swings smoothly across the photographed side instead of a full circle."""
    c, R, s = site_frame(chunk)
    X, Y, U = R[0], R[1], R[2]
    a = 0.5 * max(s[0], s[1])                                  # horizontal half-extent
    fit = a / np.tan(np.radians(fov) * 0.8)                    # distance that roughly fits the site in view
    D = distance * fit * 1.15
    full = pattern == "orbit" and arc >= 359.9
    t = np.linspace(0, 1, n_frames, endpoint=not full)
    if start_angle in (None, "auto"):
        cams = [chunk.cameras[i].center for i in chunk.aligned_cameras()]
        m = (np.mean(cams, 0) - c) if cams else X
        th0 = float(np.arctan2(m @ Y, m @ X))
    else:
        th0 = np.radians(float(start_angle))
    ground = c - U * 0.25 * s[2]                               # look slightly below the box centre

    def ring(theta, el, dist):
        el = np.radians(el)
        dist = np.asarray(dist, float)
        dist = dist[:, None] if dist.ndim else dist
        return (c + dist * (np.cos(el)[:, None] * (np.cos(theta)[:, None] * X + np.sin(theta)[:, None] * Y)
                            + np.sin(el)[:, None] * U))
    up = np.repeat(U[None], n_frames, 0)
    if pattern == "orbit":
        th = th0 + 2 * np.pi * t if full else th0 + np.radians(arc) * (_smooth(t) - 0.5)
        eye = ring(th, np.full(n_frames, elevation), D)
        tgt = np.repeat(ground[None], n_frames, 0)
    elif pattern == "helix":
        th = th0 + 3 * np.pi * t
        el = np.clip(elevation + 25, 5, 85) + (np.clip(elevation - 15, 5, 85) - np.clip(elevation + 25, 5, 85)) * _smooth(t)
        dist = D * (1.1 - 0.25 * _smooth(t))
        eye = ring(th, el, dist)
        tgt = np.repeat(ground[None], n_frames, 0)
    elif pattern == "flyover":
        long_ax, short_ax = (X, Y) if s[0] >= s[1] else (Y, X)
        L = 0.5 * max(s[0], s[1]) * 1.6 * distance
        h = max(D * np.sin(np.radians(max(elevation, 10))), 0.6 * a)
        u = _smooth(t)
        along = -L + 2 * L * u
        lead = h / np.tan(np.radians(40))                      # look ~40 deg down, ahead of the camera
        a_long = 0.5 * max(s[0], s[1])
        look = np.clip(along + lead, -0.7 * a_long, 0.7 * a_long)   # never look past the end of the site
        eye = c + along[:, None] * long_ax + h * U
        tgt = c + look[:, None] * long_ax - 0.25 * s[2] * U
    elif pattern == "reveal":
        u = _smooth(t)
        th = th0 + np.radians(90) * u
        el = 8 + (max(elevation, 20) - 8) * u
        dist = D * (0.25 + 0.95 * u)
        eye = ring(th, el, dist)
        tgt = np.repeat(ground[None], n_frames, 0)
    elif pattern == "topdown":
        u = _smooth(t)
        th = th0 + np.radians(120) * t
        h = D * (1.5 - 0.5 * u)
        eye = c + h[:, None] * U + 1e-3 * D * X                 # tiny offset: avoid a degenerate view-up
        tgt = np.repeat(c[None], n_frames, 0)
        up = np.cos(th)[:, None] * Y + np.sin(th)[:, None] * X
    elif pattern == "capture":
        from . import camera as cm
        cams = [i for i in chunk.aligned_cameras()]
        if len(cams) < 2:
            raise RuntimeError("need aligned cameras for the 'capture' path")
        cams.sort(key=lambda i: (str(chunk.cameras[i].source or ""), chunk.cameras[i].frame or 0, chunk.cameras[i].label))
        C = np.array([chunk.cameras[i].center for i in cams])
        F = np.array([cm.rodrigues_to_R(chunk.cameras[i].rvec)[0].T @ [0, 0, 1.0] for i in cams])
        Uc = np.array([-(cm.rodrigues_to_R(chunk.cameras[i].rvec)[0].T @ [0, 1.0, 0]) for i in cams])
        from scipy.ndimage import gaussian_filter1d
        sig = max(1.0, len(cams) / 40)
        C = gaussian_filter1d(C, sig, axis=0, mode="nearest")
        F = gaussian_filter1d(F, sig * 1.5, axis=0, mode="nearest")
        Uc = gaussian_filter1d(Uc, sig * 1.5, axis=0, mode="nearest")
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
        q = np.linspace(0, seg[-1], n_frames)
        eye = np.stack([np.interp(q, seg, C[:, k]) for k in range(3)], 1)
        fwd = np.stack([np.interp(q, seg, F[:, k]) for k in range(3)], 1)
        fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
        dist = np.median(np.linalg.norm(C - c, axis=1))
        tgt = eye + fwd * dist
        up = np.stack([np.interp(q, seg, Uc[:, k]) for k in range(3)], 1)
    else:
        raise ValueError("unknown pattern %s (choose from %s)" % (pattern, ", ".join(PATTERNS)))
    return eye, tgt, up


def _textured_polydata(pv, V, F, uv, face_uv):
    cu = uv[face_uv.ravel()]
    q = np.round(cu * 2 ** 20).astype(np.int64)
    key = np.stack([F.ravel().astype(np.int64), q[:, 0], q[:, 1]], 1)
    uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    m = pv.PolyData(V[uniq[:, 0]].astype(np.float32), np.hstack([np.full((len(F), 1), 3), inv.reshape(-1, 3)]).ravel())
    tc = cu[first].astype(np.float32)
    try:
        m.active_texture_coordinates = tc
    except AttributeError:
        m.t_coords = tc
    return m


def _scene(pl, chunk, display):
    import pyvista as pv
    from .io import read_ply
    d = chunk.dir
    mesh_p, cloud_p = os.path.join(d, "mesh.ply"), os.path.join(d, "point_cloud.ply")
    uvp, tex = os.path.join(d, "model", "uv.npz"), os.path.join(d, "model", "model_texture.jpg")
    if display == "auto":
        display = ("textured" if os.path.exists(uvp) and os.path.exists(tex) else
                   "colours" if os.path.exists(mesh_p) else "points")
    if display in ("textured", "colours", "shaded"):
        if not os.path.exists(mesh_p):
            raise RuntimeError("no mesh yet - build the mesh first or use display=points")
        r = read_ply(mesh_p)
        V, F = r["verts"], r["faces"]
        if display == "textured":
            u = np.load(uvp)
            if len(u["face_uv"]) != len(F):
                raise RuntimeError("the texture is older than the mesh - rebuild the texture")
            pl.add_mesh(_textured_polydata(pv, V, F, u["uv"], u["face_uv"]), texture=pv.read_texture(tex), lighting=False)
        else:
            m = pv.PolyData(V.astype(np.float32), np.hstack([np.full((len(F), 1), 3), F]).ravel())
            if display == "colours" and r.get("colors") is not None:
                m["rgb"] = r["colors"]
                pl.add_mesh(m, scalars="rgb", rgb=True, lighting=False)
            else:
                pl.add_mesh(m, color="#d9d4c7", smooth_shading=True, specular=0.15)
    else:
        if os.path.exists(cloud_p):
            from .io import read_ply_sampled
            r = read_ply_sampled(cloud_p, 8_000_000); P, C = r["verts"], r.get("colors")
        else:
            sp = chunk.load_sparse(); P, C = sp["points"], sp["colors"]
        step = max(1, len(P) // 6_000_000)
        kw = dict(scalars=C[::step], rgb=True) if C is not None else dict(color="#ddd")
        pl.add_points(pv.PolyData(np.asarray(P[::step], np.float32)), point_size=2, **kw)
    return display


def render_flythrough(chunk, out=None, pattern="orbit", seconds=20.0, fps=30, size="1080p", elevation=30.0,
                      distance=1.0, display="auto", start_angle="auto", crf=18, fov=50.0, arc=360.0, progress=True):
    """Render a fly-through MP4 of the chunk's model. Returns the video path."""
    import pyvista as pv
    from .video import ffmpeg_bin
    t0 = time.time()
    W, H = SIZES[size] if isinstance(size, str) else size
    W, H = int(W) // 2 * 2, int(H) // 2 * 2
    n = max(2, int(round(seconds * fps)))
    out = out or os.path.join(chunk.project.dir, "exports", "flythrough_%s.mp4" % pattern)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    eye, tgt, up = camera_path(chunk, pattern, n, elevation, distance, start_angle, fov, arc)
    pl = pv.Plotter(off_screen=True, window_size=[W, H])
    try:
        pl.set_background("#9fb4c8", top="#2b3a4c")
    except TypeError:
        pl.set_background("#3a4656")
    for aa in ("ssaa", "msaa", "fxaa"):
        try:
            pl.enable_anti_aliasing(aa); break
        except Exception:
            continue
    shown = _scene(pl, chunk, display)
    pl.camera.view_angle = fov
    log("Fly-through: %s, %d frames %dx%d @ %d fps, display %s -> %s" % (pattern, n, W, H, fps, shown, out))
    cmd = [ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    poster = None
    try:
        for k in range(n):
            cam = pl.camera
            cam.position = tuple(eye[k]); cam.focal_point = tuple(tgt[k]); cam.up = tuple(up[k])
            pl.reset_camera_clipping_range()
            pl.render()
            img = pl.screenshot(return_img=True)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            proc.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())
            if k == n // 3:
                poster = img.copy()
            if progress and k % max(1, n // 10) == 0:
                log("  frame %d/%d" % (k + 1, n))
    finally:
        proc.stdin.close()
        rc = proc.wait()
        pl.close()
    if rc != 0:
        raise RuntimeError("ffmpeg failed (exit %d)" % rc)
    if poster is not None:
        import cv2
        cv2.imwrite(out[:-4] + ".jpg", poster[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    dt = time.time() - t0
    chunk.add_log("Fly-through Video", dt, pattern=pattern, frames=n, size="%dx%d" % (W, H), file=os.path.basename(out))
    log("Fly-through written: %s (%.1f MB, %.0f s, %.1f frames/s)" % (out, os.path.getsize(out) / 1e6, dt, n / dt))
    return out


def run_external(project_dir, chunk_idx, out, **opts):
    """Render in a separate process (VTK off-screen rendering must not share the GUI's OpenGL thread).
    Streams the child's log lines into this process's log."""
    if getattr(sys, "frozen", False):
        cli = os.path.join(os.path.dirname(sys.executable), "shapekit-cli.exe")
        cmd = [cli if os.path.exists(cli) else sys.executable] + ([] if os.path.exists(cli) else ["--cli"])
        cwd = None
    else:
        cmd = [sys.executable, "-u", "-m", "shapekit"]
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd += ["flythrough", project_dir, "--chunk", str(chunk_idx), "--out", out]
    for k, v in opts.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-30:]
        if line:
            log("  " + line.split("] ", 1)[-1] if line.startswith("[") else "  " + line)
    if p.wait() != 0 or not os.path.exists(out):
        raise RuntimeError("fly-through rendering failed:\n" + "\n".join(tail[-12:]))
    return out
