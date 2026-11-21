"""Make the README gallery from a finished project: stills, photo-vs-model comparisons, fly-through videos + GIFs
and a GUI screenshot.

    uv run python docs\\make_media.py projects\\bariniki            (writes docs\\media\\)

Rendering is off-screen with VTK (GPU); videos and GIFs with FFmpeg.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from shapekit.project import Project                      # noqa: E402
from shapekit import camera as cm                          # noqa: E402
from shapekit import flythrough as FT                      # noqa: E402
from shapekit.util import log                              # noqa: E402


def plotter(W, H, plain=False):
    import pyvista as pv
    pl = pv.Plotter(off_screen=True, window_size=[W, H])
    try:
        if plain:
            raise TypeError
        pl.set_background("#9fb4c8", top="#2b3a4c")
    except TypeError:
        pl.set_background("#3a3f46")
    for aa in ("ssaa", "msaa", "fxaa"):
        try:
            pl.enable_anti_aliasing(aa); break
        except Exception:
            continue
    return pl


def snap(pl, eye, tgt, up, fov=50.0):
    cam = pl.camera
    cam.position, cam.focal_point, cam.up = tuple(eye), tuple(tgt), tuple(up)
    cam.view_angle = fov
    pl.reset_camera_clipping_range()
    pl.render()
    img = pl.screenshot(return_img=True)
    return np.ascontiguousarray(img[:, :, :3][:, :, ::-1])            # BGR for OpenCV


def add_cameras(pl, chunk, size):
    """Camera frustums as small pyramids (Metashape-style)."""
    import pyvista as pv
    pts, lines = [], []
    for i in chunk.aligned_cameras():
        c = chunk.cameras[i]; sen = chunk.sensors[c.sensor]
        R = cm.rodrigues_to_R(c.rvec)[0]; C = c.center
        f = sen.intr[0]; w, h = sen.width / f / 2, sen.height / f / 2
        corners = [C + size * (R.T @ np.array([x, y, 1.0])) for x, y in ((-w, -h), (w, -h), (w, h), (-w, h))]
        b = len(pts)
        pts += [C] + corners
        for k in range(4):
            lines += [2, b, b + 1 + k, 2, b + 1 + k, b + 1 + (k + 1) % 4]
    poly = pv.PolyData(np.array(pts, np.float32)); poly.lines = np.array(lines)
    pl.add_mesh(poly, color="#ffb000", line_width=1)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), (24, 28, 34), -1)
    cv2.putText(out, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def camera_view(chunk, q, back=1.5):
    """A viewpoint based on a real photo (fraction q along the capture): pulled back along its viewing direction
    and raised, looking at the surface it photographed, with the site's 'up'. Reconstructions are best where the
    photos looked, so these frame the model at its best."""
    c, Rs, s = FT.site_frame(chunk)
    U = Rs[2]
    cams = chunk.aligned_cameras()
    i = cams[int(q * (len(cams) - 1))]
    cam = chunk.cameras[i]; R = cm.rodrigues_to_R(cam.rvec)[0]; C = cam.center
    sp = chunk.load_sparse()
    o = sp["obs_cam"] == i
    d = np.median((sp["points"][sp["obs_pt"][o]] @ R.T + cam.tvec)[:, 2]) if o.any() else 0.1 * max(s)
    fwd = R.T @ [0, 0, 1.0]
    tgt = C + fwd * d
    eye = tgt - fwd * d * back + U * d * back * 0.5
    return eye, tgt, U


def stills(chunk, out, W=1280, H=720):
    """Pipeline stages from one viewpoint (tie points + cameras, dense cloud, shaded mesh, textured mesh) and
    textured / shaded hero shots, all from viewpoints derived from the photos (see camera_view)."""
    import pyvista as pv
    c, R, s = FT.site_frame(chunk)
    stage_view = camera_view(chunk, 0.4357, 2.2)
    hero = [camera_view(chunk, q, 1.5) for q in (0.05, 0.5643, 0.8214)]
    files = {}
    stages = []
    for name, title in (("tie", "1. Tie points + aligned cameras"), ("points", "2. Dense point cloud"),
                        ("shaded", "3. Mesh (geometry)"), ("textured", "4. Textured model")):
        pl = plotter(W, H, plain=True)
        if name == "tie":
            sp = chunk.load_sparse()
            pl.add_points(pv.PolyData(sp["points"].astype(np.float32)), scalars=sp["colors"], rgb=True, point_size=3)
            add_cameras(pl, chunk, 0.015 * max(s[0], s[1]))
        else:
            FT._scene(pl, chunk, name)
        stages.append(label(snap(pl, *stage_view), title))
        if name == "textured":
            for k, v in enumerate(hero):
                p = os.path.join(out, "textured_%d.jpg" % (k + 1))
                cv2.imwrite(p, snap(pl, *v), [cv2.IMWRITE_JPEG_QUALITY, 90]); files.setdefault(name, []).append(p)
        if name == "shaded":
            p = os.path.join(out, "shaded_1.jpg")
            cv2.imwrite(p, snap(pl, *hero[1]), [cv2.IMWRITE_JPEG_QUALITY, 90]); files[name] = [p]
        pl.close()
    grid = np.vstack([np.hstack(stages[:2]), np.hstack(stages[2:])])
    s_ = 1600 / grid.shape[1]
    grid = cv2.resize(grid, (1600, int(grid.shape[0] * s_)), interpolation=cv2.INTER_AREA)
    p = os.path.join(out, "pipeline_stages.jpg")
    cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 88]); files["stages"] = p
    return files


def photo_vs_model(chunk, out, n=3, width=960):
    """Original photo (undistorted) next to the textured model rendered from the same camera."""
    cams = chunk.aligned_cameras()
    picks = [cams[int(len(cams) * q)] for q in np.linspace(0.2, 0.8, n)]
    files = []
    for k, i in enumerate(picks):
        c = chunk.cameras[i]; sen = chunk.sensors[c.sensor]
        sc = width / max(sen.width, sen.height)
        img = cv2.imread(chunk.abspath(c))
        m1, m2, K = cm.undistort_maps(sen.calib(), sen.width, sen.height, sc)
        und = cv2.remap(img, m1, m2, cv2.INTER_AREA)
        H, W = und.shape[:2]
        R = cm.rodrigues_to_R(c.rvec)[0]; C = c.center
        fov = np.degrees(2 * np.arctan(H / 2 / K[1, 1]))
        pl = plotter(W, H, plain=True)                     # holes in the model show as dark grey
        FT._scene(pl, chunk, "textured")
        ren = snap(pl, C, C + R.T @ [0, 0, 1.0], -(R.T @ [0, 1.0, 0]), fov)
        pl.close()
        pair = np.hstack([label(und, "Photo"), np.full((H, 8, 3), 255, np.uint8), label(ren, "Model (same camera)")])
        p = os.path.join(out, "photo_vs_model_%d.jpg" % (k + 1))
        cv2.imwrite(p, pair, [cv2.IMWRITE_JPEG_QUALITY, 88]); files.append(p)
    return files


def animation(mp4, width=640, fps=12):
    """Animated WebP for the README (GitHub shows it inline; ~4x smaller than a GIF); GIF if no WebP encoder."""
    from shapekit.video import ffmpeg_bin
    webp = mp4[:-4] + ".webp"
    r = subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-i", mp4, "-vf", "fps=%d,scale=%d:-1:flags=lanczos" % (fps, width),
                        "-c:v", "libwebp", "-quality", "60", "-compression_level", "6", "-loop", "0", "-an", webp])
    if r.returncode == 0 and os.path.exists(webp):
        return webp
    gif = mp4[:-4] + ".gif"
    vf = ("fps=%d,scale=%d:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
          % (fps, width))
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-i", mp4, "-vf", vf, "-loop", "0", gif], check=True)
    return gif


def videos(chunk, out):
    files = {}
    for f in os.listdir(out):                                  # drop animations from earlier runs
        if f.startswith("flythrough_") and f.endswith((".gif", ".webp")):
            os.remove(os.path.join(out, f))
    for pattern, sec, kw in (("orbit", 12, dict(elevation=35, distance=1.0)), ("flyover", 10, dict(elevation=35)),
                             ("capture", 12, {})):
        mp4 = os.path.join(out, "flythrough_%s.mp4" % pattern)
        FT.render_flythrough(chunk, mp4, pattern, seconds=sec, fps=30, size="720p", crf=26, **kw)
        anim = animation(mp4)
        files[pattern] = (mp4, anim)
        log("  %s: mp4 %.1f MB, %s %.1f MB" % (pattern, os.path.getsize(mp4) / 1e6, anim[-4:], os.path.getsize(anim) / 1e6))
    return files


def gui_shot(project_dir, out):
    """Screenshot of the desktop app with the project open (own process: Qt + VTK on-screen)."""
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtWidgets import QApplication
    from shapekit.gui.app import MainWindow
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow(project_dir)
    w.resize(1600, 950); w.show()

    def pump(sec):
        t = time.time()
        while time.time() - t < sec:
            app.processEvents(); time.sleep(0.02)
    pump(2)
    w.set_display("Mesh (textured)"); pump(4)
    w.model.plotter.render(); pump(1)
    p = os.path.join(out, "gui.jpg")
    w.grab().save(p, "JPG", 88)
    w.close()
    return p


def main(project_dir, out=None):
    out = out or os.path.join(ROOT, "docs", "media")
    os.makedirs(out, exist_ok=True)
    if "--gui-shot" in sys.argv:
        return gui_shot(project_dir, out)
    p = Project.open(project_dir); ch = p.chunk()
    t = time.time()
    log("README media for %s -> %s" % (project_dir, out))
    res = dict(stills=stills(ch, out), photo_vs_model=photo_vs_model(ch, out), videos=videos(ch, out))
    log("  GUI screenshot...")
    subprocess.run([sys.executable, os.path.abspath(__file__), project_dir, out, "--gui-shot"], check=False)
    st = ch.stats
    summary = dict(project=os.path.basename(os.path.abspath(project_dir)), cameras=len(ch.cameras),
                   aligned=st["align"]["aligned"], rms_px=round(st["align"]["rms_reprojection_px"], 2),
                   tie_points=st["align"]["tie_points"], dense_points=st.get("point_cloud", {}).get("points"),
                   faces=st.get("mesh", {}).get("faces"), texture=st.get("texture", {}).get("size"),
                   steps={l["step"]: round(l["seconds"]) for l in ch.log})
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log("README media done in %.0f s: %s" % (time.time() - t, json.dumps(summary)))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(*a)
