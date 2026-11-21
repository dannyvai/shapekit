"""Learned local features + LightGlue matching (PyTorch, CUDA if available).

Methods: superpoint, aliked, disk  (detector/descriptor) + LightGlue (matcher), via the official
cvg/LightGlue package. Install with:  uv sync   (see pyproject.toml)
"""
import numpy as np

METHODS = ("superpoint", "aliked", "disk")
_CACHE = {}


def torch_available():
    try:
        import torch  # noqa: F401
        import lightglue  # noqa: F401
        return True
    except Exception:
        return False


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "CUDA: %s (torch %s)" % (torch.cuda.get_device_name(0), torch.__version__)
        return "CPU (torch %s)" % torch.__version__
    except Exception:
        return "no torch"


class LearnedFeatures:
    """Wraps extractor + LightGlue matcher. Keeps models on the GPU between calls."""

    def __init__(self, method="superpoint", max_keypoints=4096, resize=None, detection_threshold=None,
                 fp16=True):
        import torch
        from lightglue import SuperPoint, ALIKED, DISK, LightGlue
        if method not in METHODS:
            raise ValueError("unknown feature method %s" % method)
        self.torch = torch
        self.method = method
        self.dev = device()
        self.resize = resize
        kw = dict(max_num_keypoints=max_keypoints)
        if method == "superpoint":
            kw["detection_threshold"] = detection_threshold if detection_threshold is not None else 0.0005
            kw["nms_radius"] = 3
            self.extractor = SuperPoint(**kw)
        elif method == "aliked":
            kw["detection_threshold"] = detection_threshold if detection_threshold is not None else 0.01
            self.extractor = ALIKED(model_name="aliked-n16", **kw)
        else:
            self.extractor = DISK(**kw)
        self.extractor = self.extractor.eval().to(self.dev)
        self.matcher = LightGlue(features=method, depth_confidence=0.95, width_confidence=0.99,
                                 filter_threshold=0.1, mp=bool(fp16 and self.dev.type == "cuda"),
                                 flash=True).eval().to(self.dev)

    def memory_model(self):
        """(fixed bytes, bytes per pixel) of one extraction on this GPU, measured once on two synthetic images
        (peak allocated memory is linear in the pixel count for these fully-convolutional detectors)."""
        if getattr(self, "_mem_model", None) is not None or self.dev.type != "cuda":
            return getattr(self, "_mem_model", None)
        torch = self.torch
        pts = []
        for w, h in ((640, 480), (1280, 960)):
            img = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
            torch.cuda.synchronize(); base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            self.extract(img, resize=None)
            torch.cuda.synchronize()
            pts.append((w * h, torch.cuda.max_memory_allocated() - base))
        (p0, m0), (p1, m1) = pts
        per_px = max((m1 - m0) / (p1 - p0), 1.0)
        self._mem_model = (max(m0 - per_px * p0, 0.0), per_px)
        return self._mem_model

    def max_pixels(self, margin=0.7):
        """Largest image (in pixels) that fits in `margin` of the GPU memory free right now; None on CPU."""
        mm = self.memory_model()
        if mm is None:
            return None
        torch = self.torch
        free, _ = torch.cuda.mem_get_info()
        free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()     # cached blocks are reusable
        fixed, per_px = mm
        return max(int((margin * free - fixed) / per_px), 640 * 480)

    def extract(self, img_bgr, resize="default"):
        """img_bgr uint8 HxWx3 -> dict(xy (N,2) float32 in full-res pixel coords, desc (N,D) float16, score).
        resize: longest side the network sees (None = native); "default" = the value given at construction."""
        if resize == "default":
            resize = self.resize
        torch = self.torch
        rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
        t = torch.from_numpy(rgb).to(self.dev).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            f = self.extractor.extract(t, resize=resize)
        return dict(xy=f["keypoints"][0].float().cpu().numpy(),
                    desc=f["descriptors"][0].float().cpu().numpy().astype(np.float16),
                    score=f["keypoint_scores"][0].float().cpu().numpy())

    def _to_dev(self, xy, desc, size):
        torch = self.torch
        return {"keypoints": torch.from_numpy(np.asarray(xy, np.float32))[None].to(self.dev),
                "descriptors": torch.from_numpy(np.asarray(desc, np.float16)).to(self.dev).float()[None],
                "image_size": torch.tensor([size], dtype=torch.float32, device=self.dev)}

    def to_device(self, key, xy, desc, size):
        """Upload once; cached on the GPU (LRU) so each image is transferred a single time."""
        cache = self.__dict__.setdefault("_dev_cache", {})
        if key in cache:
            return cache[key]
        if len(cache) > 400:
            cache.pop(next(iter(cache)))
        cache[key] = self._to_dev(xy, desc, size)
        return cache[key]

    def match_dev(self, d0, d1):
        torch = self.torch
        if d0["keypoints"].shape[1] < 2 or d1["keypoints"].shape[1] < 2:
            return np.zeros((0, 2), int), np.zeros(0)
        with torch.inference_mode():
            out = self.matcher({"image0": d0, "image1": d1})
        return out["matches"][0].cpu().numpy().astype(int), out["scores"][0].float().cpu().numpy()

    def match(self, f0, f1, size0, size1):
        """f = (xy, desc). Returns (K,2) index pairs and (K,) scores."""
        torch = self.torch
        if len(f0[0]) < 2 or len(f1[0]) < 2:
            return np.zeros((0, 2), int), np.zeros(0)
        with torch.inference_mode():
            out = self.matcher({"image0": self._to_dev(*f0, size0), "image1": self._to_dev(*f1, size1)})
        m = out["matches"][0].cpu().numpy().astype(int)
        s = out["scores"][0].float().cpu().numpy()
        return m, s


def get(method, max_keypoints, resize):
    key = (method, max_keypoints, resize)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = LearnedFeatures(method, max_keypoints, resize)
    return _CACHE[key]
