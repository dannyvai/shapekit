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
os.makedirs(os.path.join(dist, "ffmpeg"), exist_ok=True)
for name in ("ffmpeg", "ffprobe"):
    p = shutil.which(name)
    if p:
        shutil.copy2(os.path.realpath(p), os.path.join(dist, "ffmpeg", name + ".exe"))
        print("bundled", name, "from", os.path.realpath(p))
    else:
        print("WARNING: %s not found on PATH - video import in the portable build needs it next to ShapeKit.exe" % name)
with open(os.path.join(dist, "README.txt"), "w", encoding="utf-8") as f:
    f.write("""ShapeKit portable
=================
ShapeKit.exe              desktop app (double-click). ShapeKit.exe <project folder> opens that project.
shapekit-cli.exe ...      command line, e.g.  shapekit-cli.exe process D:\\site\\proj --video D:\\site\\*.mp4
shapekit-cli.exe doctor   checks GPU, models, FFmpeg and the GUI

Copy the whole ShapeKit folder anywhere (USB disk, another PC) - nothing to install.
GPU: any NVIDIA card with a recent driver (CUDA 12.8 runtime is included; driver 570 or newer).
Without an NVIDIA GPU it still runs, on the CPU (much slower).
Logs of the desktop app: %LOCALAPPDATA%\\ShapeKit\\shapekit.log
models\\ holds the network weights (features, matching, retrieval, AI masks): no internet needed.
""")
total = 0
for r, _, fs in os.walk(dist):
    total += sum(os.path.getsize(os.path.join(r, x)) for x in fs)
print("dist\\ShapeKit: %.2f GB" % (total / 1e9))
