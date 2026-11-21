import sys
import time
import numpy as np
import cv2

_T0 = time.time()


LOG_HOOKS = []      # GUI console subscribes here


def log(msg):
    line = "[%7.1fs] %s" % (time.time() - _T0, msg)
    print(line, flush=True)
    for h in list(LOG_HOOKS):
        try:
            h(line)
        except Exception:
            pass


class Timer:
    def __enter__(self):
        self.t = time.time(); return self

    def __exit__(self, *a):
        self.s = time.time() - self.t


def read_image(path, scale=1.0, gray=False):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise IOError("cannot read %s" % path)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return img


class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find_all(self):
        p = self.p
        while True:
            q = p[p]
            if np.array_equal(q, p):
                return p
            p = q

    def union_many(self, a, b):
        """Vectorised-ish union: iterate pointer jumping until stable."""
        a = np.asarray(a); b = np.asarray(b)
        for _ in range(100):
            ra, rb = self.find_all()[a], self.find_all()[b]
            diff = ra != rb
            if not diff.any():
                break
            lo, hi = np.minimum(ra[diff], rb[diff]), np.maximum(ra[diff], rb[diff])
            # link larger root to smaller root (min wins on conflicts)
            np.minimum.at(self.p, hi, lo)
            self.p = self.find_all()
        return self.p


import os as _os


def fast_mode():
    """Optimised (parallel / GPU / fast I/O) code paths. Set SHAPEKIT_LEGACY=1 to run the original
    single-threaded implementations (used by the benchmark to verify identical results)."""
    return not _os.environ.get("SHAPEKIT_LEGACY")


def n_workers(cap=12):
    return max(1, min(cap, (_os.cpu_count() or 2) - 1))
