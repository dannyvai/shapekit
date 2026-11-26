import sys
import time
import numpy as np
import cv2

_T0 = time.time()


# ---- Unicode safety (Windows) ------------------------------------------------------------------------------
# 1) The console / redirected log uses the ANSI code page (cp1252): printing a Hebrew/Cyrillic/... path raised
#    UnicodeEncodeError. Write UTF-8 instead; characters a real console cannot show become '?'.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# 2) OpenCV's imread/imwrite cannot open non-ASCII paths on Windows (they return None / False). Route such
#    paths through Python file I/O + imdecode/imencode. Installed once for every module that uses cv2.
def _install_unicode_imageio():
    if getattr(cv2, "_shapekit_unicode", False):
        return
    _imread, _imwrite = cv2.imread, cv2.imwrite

    def imread(path, flags=cv2.IMREAD_COLOR):
        p = str(path)
        if p.isascii():
            return _imread(p, flags)
        try:
            data = np.fromfile(p, np.uint8)
        except OSError:
            return None
        return cv2.imdecode(data, flags) if data.size else None

    def imwrite(path, img, params=None):
        p = str(path)
        if p.isascii():
            return _imwrite(p, img, params) if params is not None else _imwrite(p, img)
        ext = os.path.splitext(p)[1] or ".png"
        ok, buf = cv2.imencode(ext, img, params if params is not None else [])
        if not ok:
            return False
        try:
            buf.tofile(p)
        except OSError:
            return False
        return True
    cv2.imread, cv2.imwrite = imread, imwrite
    cv2._shapekit_unicode = True


import os                                                         # noqa: E402
_install_unicode_imageio()


LOG_HOOKS = []      # GUI console subscribes here


def log(msg):
    line = "[%7.1fs] %s" % (time.time() - _T0, msg)
    try:
        print(line, flush=True)
    except (UnicodeEncodeError, ValueError, OSError):           # odd stream encoding / closed console
        try:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            pass
    for h in list(LOG_HOOKS):
        try:
            h(line)
        except Exception:
            pass


def available_memory():
    """Bytes of RAM currently available to this process (free + reclaimable); None if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.ullAvailPhys)
        with open("/proc/meminfo") as f:
            info = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in f}
        return info.get("MemAvailable", info.get("MemFree"))
    except Exception:
        return None


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
