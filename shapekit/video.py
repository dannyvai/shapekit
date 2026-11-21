"""Import Video / Add Photos / Estimate Image Quality.

Video import selects keyframes automatically (like Metashape's Import Video with frame step,
but motion-adaptive): frames are tracked with optical flow and a new keyframe is taken once the
scene has moved by `overlap_shift` of the image width; within each window the sharpest frame wins.
"""
import glob
import json
import os
import shutil
import subprocess
import numpy as np
import cv2

from .project import Sensor, Camera
from .util import log

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def ffmpeg_bin(name="ffmpeg"):
    # portable build: ffmpeg.exe / ffprobe.exe shipped next to ShapeKit.exe (or in its ffmpeg\ folder)
    import sys
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
        for d in (base, os.path.join(base, "ffmpeg")):
            for n in (name + ".exe", name):
                if os.path.isfile(os.path.join(d, n)):
                    return os.path.join(d, n)
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError("%s not found on PATH. Install FFmpeg (https://ffmpeg.org) to import videos." % name)
    return exe


def probe(path):
    out = subprocess.run([ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:stream_side_data=rotation:stream_tags=rotate",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if "tags" in s and "rotate" in s["tags"]:
        rot = int(s["tags"]["rotate"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = (s.get("avg_frame_rate") if s.get("avg_frame_rate", "0/0") != "0/0" else s.get("r_frame_rate", "30/1")).split("/")
    fps = float(num) / max(float(den), 1)
    n = int(s.get("nb_frames") or 0) or int(float(s.get("duration", 0)) * fps)
    return dict(width=w, height=h, fps=fps, frames=n)


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def select_keyframes(path, overlap_shift=0.08, max_gap_s=1.5, analysis_width=640, max_fps=30.0, progress=True):
    """Return list of (frame_index, sharpness) for selected keyframes."""
    info = probe(path)
    aw = analysis_width
    ah = int(round(info["height"] * aw / info["width"] / 2)) * 2
    step = max(1, int(round(info["fps"] / max_fps)))
    cmd = [ffmpeg_bin(), "-v", "error", "-i", path, "-vf", "scale=%d:%d,format=gray" % (aw, ah),
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=aw * ah * 8)
    fsize = aw * ah
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    selected, window = [], []
    prev, p0, ref_pts, n = None, None, None, -1
    thresh = overlap_shift * aw
    max_gap = int(max_gap_s * info["fps"])
    last_key = -10 ** 9
    while True:
        buf = proc.stdout.read(fsize)
        if len(buf) < fsize:
            break
        n += 1
        if n % step:
            continue
        g = np.frombuffer(buf, np.uint8).reshape(ah, aw)
        s = sharpness(g)
        if prev is None:
            selected.append((n, s)); last_key = n
            p0 = cv2.goodFeaturesToTrack(g, 500, 0.01, 8)
            ref_pts = p0.copy() if p0 is not None else None
            prev = g
            continue
        motion, frac = thresh * 10, 0.0
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, **lk)
            ok = st.ravel() == 1
            frac = ok.sum() / max(len(ref_pts), 1)
            if ok.sum() >= 8:
                motion = float(np.median(np.linalg.norm((p1 - ref_pts)[ok].reshape(-1, 2), axis=1)))
                p0, ref_pts = p1[ok].reshape(-1, 1, 2), ref_pts[ok].reshape(-1, 1, 2)
        if motion > 0.6 * thresh or n - last_key > 0.6 * max_gap:
            window.append((n, s))
        if motion >= thresh or frac < 0.35 or n - last_key >= max_gap:
            if window:
                best = max(window, key=lambda x: x[1])
                selected.append(best); last_key = best[0]
            window = []
            p0 = cv2.goodFeaturesToTrack(g, 500, 0.01, 8)
            ref_pts = p0.copy() if p0 is not None else None
        prev = g
        if progress and n % (step * 300) == 0:
            log("  analysed %d/%d frames, %d keyframes" % (n, info["frames"], len(selected)))
    proc.stdout.close(); proc.wait()
    if window:
        selected.append(max(window, key=lambda x: x[1]))
    # de-duplicate, sort
    seen, out = set(), []
    for f, s in sorted(selected):
        if f not in seen:
            seen.add(f); out.append((f, s))
    return out, info


def extract_frames(path, frames, out_dir, prefix, quality=2):
    """Extract given frame indices at full resolution as JPEG."""
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_tmp_extract")
    os.makedirs(tmp, exist_ok=True)
    names = []
    for b in range(0, len(frames), 150):
        batch = frames[b:b + 150]
        expr = "+".join("eq(n\\,%d)" % f for f in batch)
        cmd = [ffmpeg_bin(), "-v", "error", "-y", "-i", path, "-vf", "select='%s'" % expr,
               "-vsync", "0", "-q:v", str(quality), os.path.join(tmp, "x_%06d.jpg")]
        subprocess.run(cmd, check=True)
        outs = sorted(glob.glob(os.path.join(tmp, "x_*.jpg")))
        if len(outs) != len(batch):
            log("  warning: expected %d frames, got %d" % (len(batch), len(outs)))
        for f, o in zip(batch, outs):
            name = "%s_%06d.jpg" % (prefix, f)
            os.replace(o, os.path.join(out_dir, name))
            names.append((f, name))
        for o in glob.glob(os.path.join(tmp, "x_*.jpg")):
            os.remove(o)
    os.rmdir(tmp)
    return names


def get_sensor(chunk, w, h, label):
    for i, s in enumerate(chunk.sensors):
        if s.label == label and s.width == w and s.height == h:
            return i
    chunk.sensors.append(Sensor(label, w, h))
    return len(chunk.sensors) - 1


def import_video(chunk, path, overlap_shift=0.08, max_gap_s=1.5, shared_sensor=True):
    """Import keyframes from a video into the chunk. All frames of one device/resolution share a sensor."""
    stem = os.path.splitext(os.path.basename(path))[0]
    log("Import video %s" % path)
    keys, info = select_keyframes(path, overlap_shift, max_gap_s)
    log("  %d keyframes selected from %d frames (%.0f fps)" % (len(keys), info["frames"], info["fps"]))
    sharp = dict(keys)
    names = extract_frames(path, [k for k, _ in keys], os.path.join(chunk.dir, "images"), stem)
    sensor = get_sensor(chunk, info["width"], info["height"],
                        "video %dx%d" % (info["width"], info["height"]) if shared_sensor else stem)
    vals = np.array([sharp[f] for f, _ in names]) if names else np.array([1.0])
    ref = np.percentile(vals, 90) if len(vals) else 1.0
    for f, name in names:
        chunk.cameras.append(Camera(os.path.splitext(name)[0], os.path.join("images", name), sensor,
                                    quality=round(min(1.0, sharp[f] / max(ref, 1e-6)), 3),
                                    source=os.path.basename(path), frame=f))
    return len(names)


def add_photos(chunk, paths, copy=True):
    """Add still photos. Images with equal resolution share a sensor (one camera body assumed)."""
    n = 0
    for p in paths:
        if os.path.isdir(p):
            add_photos(chunk, sorted(q for q in glob.glob(os.path.join(p, "*")) if q.lower().endswith(IMG_EXT)), copy)
            continue
        img = cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img is None:
            log("  skip unreadable %s" % p); continue
        full = cv2.imread(p, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        h, w = full.shape[:2]
        name = os.path.basename(p)
        rel = os.path.join("images", name)
        if copy:
            os.makedirs(os.path.join(chunk.dir, "images"), exist_ok=True)
            shutil.copy2(p, os.path.join(chunk.dir, rel))
        else:
            rel = os.path.abspath(p)
        sensor = get_sensor(chunk, w, h, "photo %dx%d" % (w, h))
        chunk.cameras.append(Camera(os.path.splitext(name)[0], rel, sensor, quality=sharpness(img)))
        n += 1
    # normalise photo quality per sensor
    return n


def estimate_image_quality(chunk, disable_below=0.0):
    """Metashape-style Estimate Image Quality: relative sharpness in [0,1] per sensor; optionally disable."""
    raw = []
    for c in chunk.cameras:
        g = cv2.imread(chunk.abspath(c), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        raw.append(sharpness(g) if g is not None else 0.0)
    raw = np.array(raw)
    sens = np.array([c.sensor for c in chunk.cameras])
    for s in np.unique(sens):
        m = sens == s
        ref = np.percentile(raw[m], 90)
        for i in np.where(m)[0]:
            chunk.cameras[i].quality = round(float(min(1.0, raw[i] / max(ref, 1e-6))), 3)
            if chunk.cameras[i].quality < disable_below:
                chunk.cameras[i].enabled = False
    return raw
