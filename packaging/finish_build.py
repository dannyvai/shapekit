"""Post-build: copy model weights, FFmpeg and a README into dist\\ShapeKit, report size."""
import os
import shutil
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dist = os.path.join(root, "dist", "ShapeKit")
models = os.path.join(root, "models")
if not os.path.isdir(dist):
    sys.exit("dist\\ShapeKit not found - did PyInstaller succeed?")
if os.path.isdir(models):
    shutil.copytree(models, os.path.join(dist, "models"), dirs_exist_ok=True)
    print("models copied")
# FFmpeg: use the one shapekit itself would find (saved setting, repo ffmpeg\, PATH, winget/choco/scoop);
# if there is none, download a static build into <repo>\ffmpeg (also makes running from source work)
sys.path.insert(0, root)
from shapekit import ffmpeg as FF          # noqa: E402
if not (FF.find("ffmpeg") and FF.find("ffprobe")):
    print("FFmpeg not installed on this PC - downloading it")
    from shapekit.tools.get_ffmpeg import main as get_ffmpeg   # noqa: E402
    if get_ffmpeg(os.path.join(root, "ffmpeg")) != 0:
        sys.exit("ERROR: no FFmpeg for the portable build (see above). Install it or run: "
                 "uv run python -m shapekit.tools.get_ffmpeg")
    FF.reset_cache()
ffdir = os.path.join(dist, "ffmpeg")
os.makedirs(ffdir, exist_ok=True)
for name in ("ffmpeg", "ffprobe"):
    p = os.path.realpath(FF.ffmpeg_bin(name))
    shutil.copy2(p, os.path.join(ffdir, name + ".exe"))
    print("bundled", name, "from", p)
    for extra in ("LICENSE.txt", "SOURCE.txt"):                  # licence notes of a downloaded build
        q = os.path.join(os.path.dirname(p), extra)
        if os.path.isfile(q):
            shutil.copy2(q, os.path.join(ffdir, extra))
print(FF.version(os.path.join(ffdir, "ffmpeg.exe")))
with open(os.path.join(dist, "README.txt"), "w", encoding="utf-8") as f:
    f.write("""ShapeKit portable
=================
ShapeKit.exe              desktop app (double-click). ShapeKit.exe <project folder> opens that project.
shapekit-cli.exe ...      command line, e.g.  shapekit-cli.exe process D:\\site\\proj --video D:\\site\\*.mp4
shapekit-cli.exe doctor   checks GPU, models, FFmpeg and the GUI
shapekit-cli.exe config ffmpeg D:\\tools\\ffmpeg\\bin    use another FFmpeg (or in the app: Tools > FFmpeg Location...)

Copy the whole ShapeKit folder anywhere (USB disk, another PC) - nothing to install.
GPU: any NVIDIA card with a recent driver (CUDA 12.8 runtime is included; driver 570 or newer).
Without an NVIDIA GPU it still runs, on the CPU (much slower).
Logs of the desktop app: %LOCALAPPDATA%\\ShapeKit\\shapekit.log
ffmpeg\\ holds FFmpeg (video import, fly-through videos); see ffmpeg\\LICENSE.txt if present.
models\\ holds the network weights (features, matching, retrieval, AI masks): no internet needed.
""")
total = 0
for r, _, fs in os.walk(dist):
    total += sum(os.path.getsize(os.path.join(r, x)) for x in fs)
print("dist\\ShapeKit: %.2f GB" % (total / 1e9))
