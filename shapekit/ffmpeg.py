"""Finding FFmpeg / FFprobe.

Search order (first hit wins; ffprobe is looked up the same way, preferring ffmpeg's own folder):
  1. SHAPEKIT_FFMPEG environment variable   (ffmpeg.exe or the folder holding it)
  2. the saved setting  "ffmpeg"            (GUI: Tools > FFmpeg Location..., CLI: shapekit config ffmpeg PATH)
  3. bundled copy: <ShapeKit.exe folder>[\\ffmpeg]  or, from source, <repo>\\ffmpeg
  4. PATH
  5. usual install folders (winget, Chocolatey, Scoop, C:\\ffmpeg\\bin, Program Files)
     - covers e.g. `winget install Gyan.FFmpeg` before a new terminal / log-off picks up the new PATH
"""
import glob
import os
import shutil
import subprocess
import sys

from . import config

EXE = ".exe" if sys.platform == "win32" else ""
ENV = "SHAPEKIT_FFMPEG"
HELP = ("Set its location in the app (Tools > FFmpeg Location...) or with  shapekit config ffmpeg <path>, "
        "download it with  shapekit get-ffmpeg, or install FFmpeg (winget install Gyan.FFmpeg) and put it on PATH.")


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # repo root


def bundled_dir():
    """Where the portable build ships FFmpeg (and where `get-ffmpeg` puts it by default)."""
    return os.path.join(app_dir(), "ffmpeg")


def _as_dir(p):
    """A user-given ffmpeg path (file or folder) -> folder that should hold ffmpeg/ffprobe."""
    if not p:
        return None
    p = os.path.expandvars(os.path.expanduser(str(p).strip().strip('"')))
    if os.path.isfile(p):
        return os.path.dirname(p)
    if os.path.isdir(p):
        sub = os.path.join(p, "bin")              # accept the unpacked release folder too
        if not os.path.isfile(os.path.join(p, "ffmpeg" + EXE)) and os.path.isfile(os.path.join(sub, "ffmpeg" + EXE)):
            return sub
        return p
    return None


def _well_known_dirs():
    if sys.platform != "win32":
        return ["/usr/local/bin", "/opt/homebrew/bin", "/usr/bin"]
    la = os.environ.get("LOCALAPPDATA", "")
    pf = [os.environ.get(k) for k in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432")]
    out = [os.path.join(la, "Microsoft", "WinGet", "Links")]
    # winget's package folder, e.g. ...\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin
    for pkgs in (os.path.join(la, "Microsoft", "WinGet", "Packages"),
                 *[os.path.join(p, "WinGet", "Packages") for p in pf if p]):
        out += sorted(glob.glob(os.path.join(pkgs, "*FFmpeg*", "*", "bin")), reverse=True)
    out += [r"C:\ffmpeg\bin", r"C:\ffmpeg"]
    out += [os.path.join(p, "ffmpeg", "bin") for p in pf if p]
    out += [os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), "chocolatey", "bin"),
            os.path.join(os.path.expanduser("~"), "scoop", "shims")]
    return out


def candidates():
    """(source, folder) pairs in search order; folder None = 'look on PATH'."""
    out = []
    if os.environ.get(ENV):
        out.append(("%s environment variable" % ENV, _as_dir(os.environ[ENV])))
    if config.get("ffmpeg"):
        out.append(("saved setting", _as_dir(config.get("ffmpeg"))))
    out.append(("bundled", app_dir()))
    out.append(("bundled", bundled_dir()))
    out.append(("PATH", None))
    out += [("installed", d) for d in _well_known_dirs()]
    return out


_cache = {}


def find(name="ffmpeg"):
    """Full path of ffmpeg / ffprobe, or None."""
    if name in _cache and os.path.isfile(_cache[name]):
        return _cache[name]
    hit = None
    first = find("ffmpeg") if name != "ffmpeg" else None
    if first:                                     # ffprobe: prefer the folder ffmpeg came from (same version)
        p = os.path.join(os.path.dirname(first), name + EXE)
        if os.path.isfile(p):
            hit = p
    if hit is None:
        for _, d in candidates():
            if d is None:
                hit = shutil.which(name)
            elif d and os.path.isfile(os.path.join(d, name + EXE)):
                hit = os.path.join(d, name + EXE)
            if hit:
                break
    if hit:
        _cache[name] = hit
    return hit


def reset_cache():
    _cache.clear()


def ffmpeg_bin(name="ffmpeg"):
    exe = find(name)
    if exe is None:
        raise RuntimeError("%s not found. %s" % (name, HELP))
    return exe


def version(exe):
    """First line of `exe -version`; raises if it does not run."""
    out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError("%s does not run (exit %d)" % (exe, out.returncode))
    return out.stdout.splitlines()[0]


def set_location(p):
    """Validate and save a user-chosen ffmpeg (file or folder). p=None/'' clears the setting.
    Returns (ffmpeg_path, ffprobe_path) of the new effective location."""
    if p:
        d = _as_dir(p)
        if d is None:
            raise ValueError("%s does not exist" % p)
        ff, fp = os.path.join(d, "ffmpeg" + EXE), os.path.join(d, "ffprobe" + EXE)
        missing = [os.path.basename(x) for x in (ff, fp) if not os.path.isfile(x)]
        if missing:
            raise ValueError("%s not found in %s (ffmpeg and ffprobe must be in the same folder)" % (" and ".join(missing), d))
        version(ff)
        config.set("ffmpeg", d)
    else:
        config.set("ffmpeg", None)
    reset_cache()
    return find("ffmpeg"), find("ffprobe")


def describe():
    """Human-readable status, e.g. for `shapekit config` and the GUI dialog."""
    lines = []
    for name in ("ffmpeg", "ffprobe"):
        p = find(name)
        lines.append("%-8s %s" % (name, p or "NOT FOUND"))
    saved = config.get("ffmpeg")
    lines.append("saved setting: %s" % (saved or "(none)"))
    if os.environ.get(ENV):
        lines.append("%s = %s" % (ENV, os.environ[ENV]))
    p = find("ffmpeg")
    if p:
        try:
            lines.append(version(p))
        except Exception as e:
            lines.append("ERROR: %s" % e)
    return "\n".join(lines)
