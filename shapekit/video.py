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


from .ffmpeg import ffmpeg_bin, passthrough  # noqa: E402,F401  (re-exported: flythrough / doctor import it from here)


def probe(path):
    out = subprocess.run([ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:stream_side_data=rotation:stream_tags=rotate",
                          "-of", "json", path], capture_output=True, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if out.returncode != 0 or not json.loads(out.stdout or "{}").get("streams"):
        raise RuntimeError("ffprobe cannot read a video stream in %s: %s"
                           % (path, (out.stderr or "no video stream").strip()[-300:]))
    s = json.loads(out.stdout)["streams"][0]
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


def select_keyframes(path, overlap_shift=0.08, max_gap_s=1.5, analysis_width=640, max_fps=30.0, progress=True,
                     min_sharpness=0.7, sharp_width=1280):
    """Return ([(frame_index, sharpness), ...], info) for the selected keyframes.

    Motion: optical flow on a `analysis_width` copy; a keyframe is due once the view moved by `overlap_shift` of
    the image width (or after `max_gap_s` seconds / when tracking is lost).
    Blur: Laplacian variance on a larger `sharp_width` copy (fine motion blur disappears when downscaled to 640),
    compared with the sharpest frames nearby (last ~2 s while scanning, +-1 s for the final check) ("relative sharpness": 1 = as sharp as the best nearby
    frames). Content changes slowly, so this separates blur from "less texture in view".
    Selection, per window:
      - candidates are all frames between 0.35x and 1x the target motion; the sharpest one wins;
      - if even that one is below `min_sharpness` (relative), the window is extended up to 1.6x the target
        motion to look for a sharper frame (overlap stays high: 1.6 x the default 8 % shift is 13 % of the width);
      - afterwards, keyframes still below `min_sharpness` are dropped where their neighbours are close enough
        that no gap appears (<= 1.8x target motion and <= max gap apart).
    min_sharpness=0 restores plain "sharpest frame per window" behaviour."""
    from collections import deque
    info = probe(path)
    sw = min(int(sharp_width), info["width"]) // 2 * 2
    sh = int(round(info["height"] * sw / info["width"] / 2)) * 2
    aw = min(analysis_width, sw)
    ah = int(round(info["height"] * aw / info["width"] / 2)) * 2
    step = max(1, int(round(info["fps"] / max_fps)))
    import tempfile
    cmd = [ffmpeg_bin(), "-v", "error", "-i", path, "-vf", "scale=%d:%d,format=gray" % (sw, sh),
           *passthrough(), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    fsize = sw * sh
    errf = tempfile.TemporaryFile()                     # ffmpeg's messages (a file: a full pipe could block ffmpeg)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, bufsize=fsize * 4,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    thresh = overlap_shift * aw
    lo, hi = 0.35, 1.6                                  # candidate window / extension limit (x target motion)
    max_gap = int(max_gap_s * info["fps"])
    recent = deque(maxlen=max(8, int(2.0 * info["fps"] / step)))      # sharpness of the last ~2 s
    selected, window = [], []                           # items: [frame, sharpness, relative, travel]
    prev, p0, ref_pts, n = None, None, None, -1
    last_key, travel_base, first, extended = -10 ** 9, 0.0, True, 0
    all_s = []                                          # (frame, sharpness) of every analysed frame

    def reset_tracks(g):
        q = cv2.goodFeaturesToTrack(g, 500, 0.01, 8)
        return q, (q.copy() if q is not None else None)

    while True:
        buf = proc.stdout.read(fsize)
        if len(buf) < fsize:
            break
        n += 1
        if n % step:
            continue
        full = np.frombuffer(buf, np.uint8).reshape(sh, sw)
        s = sharpness(full)
        g = full if aw == sw else cv2.resize(full, (aw, ah), interpolation=cv2.INTER_AREA)
        recent.append(s); all_s.append((n, s))
        ref = max(float(np.percentile(recent, 90)), 1e-6)
        rel = s / ref
        if prev is None:
            window.append([n, s, rel, 0.0])
            p0, ref_pts = reset_tracks(g)
            prev = g
            last_key = n
            continue
        motion, frac = thresh * 10, 0.0
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, **lk)
            ok = st.ravel() == 1
            frac = ok.sum() / max(len(ref_pts), 1)
            if ok.sum() >= 8:
                motion = float(np.median(np.linalg.norm((p1 - ref_pts)[ok].reshape(-1, 2), axis=1)))
                p0, ref_pts = p1[ok].reshape(-1, 1, 2), ref_pts[ok].reshape(-1, 1, 2)
        lost = frac < 0.35 or n - last_key >= max_gap
        if first:
            # first keyframe: sharpest frame before the view has moved 40 % of the target
            window.append([n, s, rel, travel_base + min(motion, thresh)])
            close = lost or motion >= 0.4 * thresh
        else:
            if motion > lo * thresh or n - last_key > 0.6 * max_gap:
                window.append([n, s, rel, travel_base + min(motion, hi * thresh)])
            best = max(window, key=lambda x: x[1]) if window else None
            good = best is not None and best[2] >= min_sharpness
            # due (>= 1x target) and the best candidate is sharp enough -> take it; if it is blurry keep
            # looking up to `hi` x target; tracking lost / max gap -> take the best we have
            close = lost or motion >= hi * thresh or (motion >= thresh and good)
            if motion >= thresh and not close:
                extended += 1
        if close:
            moved = min(motion, hi * thresh) if motion < thresh * 10 else thresh    # thresh*10 = flow failed
            best = max(window, key=lambda x: x[1]) if window else [n, s, rel, travel_base + moved]
            selected.append(best)
            last_key = best[0]
            travel_base += moved
            window, first = [], False
            p0, ref_pts = reset_tracks(g)
        prev = g
        if progress and n % (step * 300) == 0:
            log("  analysed %d/%d frames, %d keyframes" % (n, info["frames"], len(selected)))
    proc.stdout.close(); rc = proc.wait()
    errf.seek(0); err = errf.read().decode("utf-8", "replace").strip(); errf.close()
    if n < 0:
        raise RuntimeError("ffmpeg could not decode %s (exit %d): %s" % (path, rc, err or "no frames"))
    if rc != 0 or err:
        log("  ffmpeg %s: %s" % ("error (exit %d)" % rc if rc else "warnings", err[-500:]))
    if window:
        selected.append(max(window, key=lambda x: x[1]))
    # de-duplicate, sort
    seen, keys = set(), []
    for k in sorted(selected, key=lambda x: x[0]):
        if k[0] not in seen:
            seen.add(k[0]); keys.append(k)
    # final relative sharpness against the sharpest frames within +-1 s (before AND after the frame)
    if all_s:
        fr = np.array([x[0] for x in all_s]); sv = np.array([x[1] for x in all_s])
        half = max(1, int(info["fps"]))
        for k in keys:
            m = np.abs(fr - k[0]) <= half
            k[2] = k[1] / max(float(np.percentile(sv[m], 90)), 1e-6)
    # drop keyframes that are still blurry where the neighbours overlap enough without them
    dropped = 0
    if min_sharpness > 0:
        i = 1
        while i < len(keys) - 1:
            a, k, b = keys[i - 1], keys[i], keys[i + 1]
            if k[2] < min_sharpness and b[3] - a[3] <= 1.8 * thresh and b[0] - a[0] <= max_gap:
                del keys[i]; dropped += 1
            else:
                i += 1
    rels = np.array([k[2] for k in keys]) if keys else np.zeros(1)
    log("  blur filter: windows extended %d frames, %d blurry keyframes dropped, %d kept below %.2f relative sharpness (median %.2f)"
        % (extended, dropped, int((rels < min_sharpness).sum()), min_sharpness, float(np.median(rels))))
    return [(k[0], k[1]) for k in keys], info


def select_expr(frames):
    """`select` filter expression matching the given frame numbers, as a balanced tree of sums:
    (eq(n,a)+eq(n,b))+(eq(n,c)+eq(n,d)).  FFmpeg 8 limits expressions to 100 nesting levels (a flat
    a+b+c+... chain of 150 terms fails with 'Cannot allocate memory'); the tree needs only ~log2(N)."""
    if len(frames) == 1:
        return "eq(n\\,%d)" % frames[0]
    h = len(frames) // 2
    return "(%s)+(%s)" % (select_expr(frames[:h]), select_expr(frames[h:]))


def extract_frames(path, frames, out_dir, prefix, quality=2):
    """Extract given frame indices at full resolution as JPEG."""
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_tmp_extract")
    os.makedirs(tmp, exist_ok=True)
    names = []
    for b in range(0, len(frames), 150):
        batch = frames[b:b + 150]
        expr = select_expr(batch)
        cmd = [ffmpeg_bin(), "-v", "error", "-y", "-i", path, "-vf", "select='%s'" % expr,
               *passthrough(), "-q:v", str(quality), os.path.join(tmp, "x_%06d.jpg")]
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0:
            raise RuntimeError("ffmpeg frame extraction failed (exit %d): %s" % (r.returncode, r.stderr.strip()[-500:]))
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


def import_video(chunk, path, overlap_shift=0.08, max_gap_s=1.5, shared_sensor=True, min_sharpness=0.7):
    """Import keyframes from a video into the chunk. All frames of one device/resolution share a sensor.
    min_sharpness: skip motion-blurred frames (0..1, relative to the sharpest nearby frames; 0 = off)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    log("Import video %s" % path)
    keys, info = select_keyframes(path, overlap_shift, max_gap_s, min_sharpness=min_sharpness)
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
