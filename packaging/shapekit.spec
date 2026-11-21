# PyInstaller spec for the portable ShapeKit folder (two executables sharing one runtime):
#   dist\ShapeKit\ShapeKit.exe       GUI (no console window)
#   dist\ShapeKit\shapekit-cli.exe   command line
# Build with build_portable.bat (it also adds the model weights and FFmpeg).
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

import sys
root = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, root)
# every shapekit module, found from the source tree (some are imported dynamically, e.g. shapekit.tools.doctor)
own = []
for d, _, files in os.walk(os.path.join(root, "shapekit")):
    if "__pycache__" in d:
        continue
    pkg = os.path.relpath(d, root).replace(os.sep, ".")
    own += [pkg if f == "__init__.py" else pkg + "." + f[:-3] for f in files if f.endswith(".py")]
hidden = (own + collect_submodules("lightglue") + collect_submodules("kornia")
          + collect_submodules("pyvista") + collect_submodules("pyvistaqt") + collect_submodules("vtkmodules")
          + collect_submodules("skimage.measure") + ["scipy.sparse.csgraph", "scipy.spatial", "PySide6.QtOpenGL",
                                                     "PySide6.QtOpenGLWidgets"])
datas = (collect_data_files("pyvista") + collect_data_files("lightglue") + collect_data_files("kornia")
         + collect_data_files("skimage"))
binaries = collect_dynamic_libs("vtkmodules")

a = Analysis([os.path.join(SPECPATH, "shapekit_app.py")], pathex=[root], binaries=binaries, datas=datas,
             hiddenimports=hidden, hookspath=[], runtime_hooks=[],
             excludes=["tkinter", "matplotlib.tests", "IPython", "jupyter", "notebook", "pytest", "torch.utils.tensorboard"],
             noarchive=False)
# PyInstaller also copies torch's CUDA DLLs to the top level (dependency analysis); torch loads them from
# torch\lib itself, so drop the duplicates (~3 GB)
torch_lib = set()
for b in a.binaries:
    dest = b[0].replace("\\", "/")
    if dest.startswith("torch/lib/"):
        torch_lib.add(os.path.basename(dest).lower())
a.binaries = [b for b in a.binaries
              if not ("/" not in b[0].replace("\\", "/") and os.path.basename(b[0]).lower() in torch_lib)]
pyz = PYZ(a.pure)
icon = os.path.join(SPECPATH, "shapekit.ico")
icon = icon if os.path.exists(icon) else None
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="ShapeKit", console=False, icon=icon,
          upx=False, disable_windowed_traceback=False)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="shapekit-cli", console=True, icon=icon, upx=False)
coll = COLLECT(gui, cli, a.binaries, a.datas, name="ShapeKit", upx=False)
