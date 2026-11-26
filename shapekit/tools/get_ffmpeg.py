"""Download a static Windows FFmpeg build (ffmpeg.exe + ffprobe.exe) into a folder:

    python -m shapekit.tools.get_ffmpeg [DEST]          (default: <repo or ShapeKit.exe folder>\\ffmpeg)
    shapekit get-ffmpeg [DEST] [--url URL]

That folder is on shapekit's FFmpeg search path, so nothing else needs configuring.
Used by build_portable.bat when no FFmpeg is installed on the build PC.
The default build is gyan.dev's "release essentials" (GPL, includes libx264 for fly-through videos).
"""
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from ..util import log

URLS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
]
WANT = ("ffmpeg.exe", "ffprobe.exe")


def _download(url, fobj):
    log("downloading %s" % url)
    req = urllib.request.Request(url, headers={"User-Agent": "shapekit"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fobj.write(b); done += len(b)
            pct = int(done * 100 / total) if total else -1
            if pct >= 0 and pct // 10 != last:
                last = pct // 10
                log("  %3d%%  %.0f / %.0f MB" % (pct, done / 1e6, total / 1e6))


def main(dest=None, url=None):
    if sys.platform != "win32":
        print("get-ffmpeg only downloads Windows builds; install FFmpeg with your package manager.")
        return 1
    from ..ffmpeg import bundled_dir, version, reset_cache
    dest = os.path.abspath(dest or bundled_dir())
    os.makedirs(dest, exist_ok=True)
    errors = []
    for u in ([url] if url else URLS):
        try:
            with tempfile.TemporaryFile() as tmp:
                _download(u, tmp)
                tmp.seek(0)
                with zipfile.ZipFile(tmp) as z:
                    names = z.namelist()
                    for want in WANT:
                        src = next((n for n in names if n.replace("\\", "/").endswith("/bin/" + want)
                                    or os.path.basename(n) == want), None)
                        if src is None:
                            raise RuntimeError("%s not in the archive" % want)
                        with z.open(src) as s, open(os.path.join(dest, want), "wb") as d:
                            shutil.copyfileobj(s, d)
                    lic = next((n for n in names if os.path.basename(n).upper() in ("LICENSE", "LICENSE.TXT")), None)
                    if lic:
                        with z.open(lic) as s, open(os.path.join(dest, "LICENSE.txt"), "wb") as d:
                            shutil.copyfileobj(s, d)
                    with open(os.path.join(dest, "SOURCE.txt"), "w", encoding="utf-8") as f:
                        f.write("FFmpeg downloaded from %s\nSource code: https://ffmpeg.org/download.html\n" % u)
            reset_cache()
            log("FFmpeg installed into %s" % dest)
            log(version(os.path.join(dest, "ffmpeg.exe")))
            return 0
        except Exception as e:
            errors.append("%s: %s" % (u, e))
            log("  failed: %s" % e)
    log("FFmpeg download failed:\n  " + "\n  ".join(errors))
    return 1


if __name__ == "__main__":
    a = sys.argv[1:]
    url = None
    if "--url" in a:
        i = a.index("--url"); url = a[i + 1]; del a[i:i + 2]
    sys.exit(main(a[0] if a else None, url))
