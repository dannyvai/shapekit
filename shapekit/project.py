"""Project / Chunk / Camera / Sensor model, saved as JSON + npz side files (like a .psx folder)."""
import json
import os
import time
import numpy as np

from . import camera as cm


class Sensor:
    def __init__(self, label, width, height, intr=None, fixed=None, model="frame"):
        self.label = label
        self.model = model or "frame"          # frame | fisheye_equidistant | fisheye_equisolid
        self.width, self.height = int(width), int(height)
        if intr is None:
            f = 0.8 * max(width, height)  # ~ 63 deg horizontal FOV default (phone video)
            intr = [f, (width - 1) / 2, (height - 1) / 2, 0, 0, 0, 0, 0]
        self.intr = np.asarray(intr, dtype=np.float64)
        self.fixed = bool(fixed) if fixed is not None else False

    def calib(self):
        """Intrinsics for projection functions (adds the model code for fisheye sensors)."""
        return cm.calib(self.intr, self.model)

    def set_model(self, model):
        """Switch projection model; resets f to a sensible start value when changing frame <-> fisheye."""
        if model == self.model:
            return
        was_fish, is_fish = self.model != "frame", model != "frame"
        if was_fish != is_fish:
            m = max(self.width, self.height)
            self.intr[0] = (0.5 if is_fish else 0.8) * m
            self.intr[3:] = 0
        self.model = model

    def to_dict(self):
        d = dict(label=self.label, width=self.width, height=self.height,
                 intr=dict(zip(cm.INTR_NAMES, map(float, self.intr[:8]))), fixed=self.fixed)
        if self.model != "frame":
            d["model"] = self.model
        return d

    @staticmethod
    def from_dict(d):
        return Sensor(d["label"], d["width"], d["height"], [d["intr"][n] for n in cm.INTR_NAMES], d.get("fixed"),
                      d.get("model", "frame"))


class Camera:
    def __init__(self, label, path, sensor, **kw):
        self.label = label
        self.path = path.replace("\\", "/")   # relative to chunk dir, always "/" separated
        self.sensor = sensor        # sensor index
        self.enabled = kw.get("enabled", True)
        self.quality = kw.get("quality")
        self.source = kw.get("source")
        self.frame = kw.get("frame")
        self.rvec = np.asarray(kw["rvec"]) if kw.get("rvec") is not None else None
        self.tvec = np.asarray(kw["tvec"]) if kw.get("tvec") is not None else None

    @property
    def aligned(self):
        return self.rvec is not None

    @property
    def center(self):
        return cm.camera_center(self.rvec, self.tvec)

    def to_dict(self):
        d = dict(label=self.label, path=self.path, sensor=self.sensor, enabled=self.enabled,
                 quality=self.quality, source=self.source, frame=self.frame)
        if self.aligned:
            d["rvec"], d["tvec"] = list(map(float, self.rvec)), list(map(float, self.tvec))
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        return Camera(d.pop("label"), d.pop("path"), d.pop("sensor"), **d)


class Chunk:
    def __init__(self, project, label, dirname):
        self.project = project
        self.label = label
        self.dirname = dirname
        self.sensors = []
        self.cameras = []
        self.region = None   # dict(center, size, R)
        self.stats = {}
        self.log = []
        self.markers = []      # see markers.py
        self.scalebars = []
        self.transform = None  # 4x4 similarity internal -> world (None = identity / arbitrary scale)

    @property
    def dir(self):
        return os.path.join(self.project.dir, self.dirname)

    def path(self, *parts):
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def abspath(self, cam):
        p = cam.path.replace("\\", "/")
        if os.path.isabs(p) or (len(p) > 1 and p[1] == ":"):
            return p
        return os.path.join(self.dir, *p.split("/"))

    def add_log(self, step, seconds, **info):
        self.log.append(dict(step=step, seconds=round(seconds, 1), time=time.strftime("%Y-%m-%d %H:%M:%S"), **info))

    def aligned_cameras(self):
        return [i for i, c in enumerate(self.cameras) if c.aligned and c.enabled]

    # sparse cloud (tie points) -------------------------------------------------
    def save_sparse(self, **arrays):
        np.savez_compressed(self.path("sparse.npz"), **arrays)

    def load_sparse(self):
        p = os.path.join(self.dir, "sparse.npz")
        return dict(np.load(p)) if os.path.exists(p) else None

    def to_dict(self):
        return dict(label=self.label, dirname=self.dirname, sensors=[s.to_dict() for s in self.sensors],
                    cameras=[c.to_dict() for c in self.cameras], region=self.region, stats=self.stats, log=self.log,
                    markers=self.markers, scalebars=self.scalebars, transform=self.transform)

    @staticmethod
    def from_dict(project, d):
        ch = Chunk(project, d["label"], d["dirname"])
        ch.sensors = [Sensor.from_dict(s) for s in d["sensors"]]
        ch.cameras = [Camera.from_dict(c) for c in d["cameras"]]
        ch.region, ch.stats, ch.log = d.get("region"), d.get("stats", {}), d.get("log", [])
        ch.markers, ch.scalebars, ch.transform = d.get("markers", []), d.get("scalebars", []), d.get("transform")
        return ch


class Project:
    FILE = "project.json"

    def __init__(self, directory):
        self.dir = os.path.abspath(directory)
        self.chunks = []

    @staticmethod
    def create(directory):
        os.makedirs(directory, exist_ok=True)
        p = Project(directory)
        p.add_chunk("Chunk 1")
        p.save()
        return p

    @staticmethod
    def open(directory):
        p = Project(directory)
        with open(os.path.join(p.dir, Project.FILE)) as f:
            d = json.load(f)
        p.chunks = [Chunk.from_dict(p, c) for c in d["chunks"]]
        return p

    def add_chunk(self, label):
        ch = Chunk(self, label, "chunk_%d" % len(self.chunks))
        os.makedirs(ch.dir, exist_ok=True)
        self.chunks.append(ch)
        return ch

    def chunk(self, i=0):
        return self.chunks[i]

    def save(self):
        tmp = os.path.join(self.dir, self.FILE + ".tmp")
        with open(tmp, "w") as f:
            json.dump(dict(app="shapekit", version=1, chunks=[c.to_dict() for c in self.chunks]), f, indent=1)
        os.replace(tmp, os.path.join(self.dir, self.FILE))
