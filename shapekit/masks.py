"""Masks (Metashape: Import Masks / Generate Masks): per-photo keep/ignore images.

Stored as chunk/masks/NNNNN.png at full photo resolution: 255 = use, 0 = ignore (black = masked, like Metashape).
Applied to tie points (observations inside a mask are dropped before alignment), depth maps (masked pixels
get no depth) and texture (masked parts of a photo are never used for colour).

AI masks: semantic segmentation (torchvision DeepLabV3, COCO/VOC classes) of people, vehicles and animals --
the things that move on a site and break matching / ghost in textures.
"""
import os
import glob
import numpy as np
import cv2

from .util import log, read_image

VOC_GROUPS = {"people": [15], "vehicles": [1, 2, 4, 6, 7, 14, 19], "animals": [3, 8, 10, 12, 13, 17]}


def mask_path(chunk, i):
    return os.path.join(chunk.dir, "masks", "%05d.png" % i)


def has_mask(chunk, i):
    return os.path.exists(mask_path(chunk, i))


def load_mask(chunk, i, size=None):
    """Bool keep-mask (True = use pixel) or None. size=(w, h) resizes (nearest)."""
    p = mask_path(chunk, i)
    if not os.path.exists(p):
        return None
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if size is not None and (m.shape[1], m.shape[0]) != tuple(size):
        m = cv2.resize(m, tuple(size), interpolation=cv2.INTER_NEAREST)
    return m >= 128


def save_mask(chunk, i, keep):
    os.makedirs(os.path.join(chunk.dir, "masks"), exist_ok=True)
    keep = np.asarray(keep, bool)
    if keep.all():
        clear_mask(chunk, i)
        return
    cv2.imwrite(mask_path(chunk, i), keep.astype(np.uint8) * 255)


def clear_mask(chunk, i):
    p = mask_path(chunk, i)
    if os.path.exists(p):
        os.remove(p)


def masked_cameras(chunk):
    return [i for i in range(len(chunk.cameras)) if has_mask(chunk, i)]


def keep_points(chunk, i, uv):
    """Bool per pixel coordinate (N,2): True where not masked (or no mask)."""
    cam = chunk.cameras[i]
    sen = chunk.sensors[cam.sensor]
    m = load_mask(chunk, i, (sen.width, sen.height))
    if m is None:
        return np.ones(len(uv), bool)
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, sen.width - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, sen.height - 1)
    return m[y, x]


def import_masks(chunk, folder, pattern="{label}_mask.png", invert=False, cameras=None):
    """Import mask images from a folder. pattern placeholders: {label} (photo name without extension),
    {filename} (with extension), {index}. Black = masked unless invert. Returns number imported."""
    n = 0
    for i in cameras if cameras is not None else range(len(chunk.cameras)):
        c = chunk.cameras[i]
        name = pattern.format(label=c.label, filename=os.path.basename(c.path), index=i)
        cands = [os.path.join(folder, name)]
        if not os.path.exists(cands[0]):
            base = os.path.splitext(name)[0]
            cands = glob.glob(os.path.join(folder, glob.escape(base) + ".*"))
        cands = [p for p in cands if os.path.exists(p)]
        if not cands:
            continue
        m = cv2.imread(cands[0], cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        sen = chunk.sensors[c.sensor]
        if (m.shape[1], m.shape[0]) != (sen.width, sen.height):
            m = cv2.resize(m, (sen.width, sen.height), interpolation=cv2.INTER_NEAREST)
        keep = m >= 128
        save_mask(chunk, i, ~keep if invert else keep)
        n += 1
    log("Imported %d masks from %s" % (n, folder))
    return n


class _Segmenter:
    def __init__(self):
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.net = deeplabv3_resnet50(weights=w).eval().to(self.dev)
        if self.dev.type == "cuda":
            self.net = self.net.half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev)[:, None, None]

    def classes(self, img_bgr, max_side=1024):
        """Per-pixel VOC class map at the image's resolution."""
        torch = self.torch
        h, w = img_bgr.shape[:2]
        s = min(1.0, max_side / max(h, w))
        small = cv2.resize(img_bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
        x = torch.from_numpy(np.ascontiguousarray(small[:, :, ::-1])).to(self.dev).permute(2, 0, 1).float() / 255
        x = ((x - self.mean) / self.std)[None]
        if self.dev.type == "cuda":
            x = x.half()
        with torch.inference_mode():
            out = self.net(x)["out"][0]
        lab = out.argmax(0).byte().cpu().numpy()
        return cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)


def generate_ai_masks(chunk, groups=("people", "vehicles"), dilate_px=12, cameras=None, keep_existing=True):
    """Mask people / vehicles / animals in the photos with a semantic segmentation network.
    keep_existing: combine with masks already present (manual edits are kept)."""
    import time
    t0 = time.time()
    ids = sorted({c for g in groups for c in VOC_GROUPS[g]})
    seg = _Segmenter()
    log("Generate masks (%s) on %s" % (", ".join(groups), "GPU" if seg.dev.type == "cuda" else "CPU"))
    cams = list(cameras) if cameras is not None else list(range(len(chunk.cameras)))
    k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)) if dilate_px else None
    n_masked = 0
    for k, i in enumerate(cams):
        img = read_image(chunk.abspath(chunk.cameras[i]))
        lab = seg.classes(img)
        hit = np.isin(lab, ids).astype(np.uint8)
        if k_ is not None and hit.any():
            hit = cv2.dilate(hit, k_)
        keep = hit == 0
        if keep_existing:
            old = load_mask(chunk, i, (img.shape[1], img.shape[0]))
            if old is not None:
                keep &= old
        save_mask(chunk, i, keep)
        n_masked += int(not keep.all())
        if k % 20 == 0:
            log("  masks %d/%d (%.1f%% of this photo masked)" % (k + 1, len(cams), 100 * (1 - keep.mean())))
    chunk.add_log("Generate Masks", time.time() - t0, groups=list(groups), masked_photos=n_masked)
    log("Masks: %d of %d photos contain masked areas" % (n_masked, len(cams)))
    return n_masked
