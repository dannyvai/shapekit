"""Environment check (also used as the smoke test of the portable build):  shapekit doctor [PROJECT]"""
import os
import sys
import time
import traceback


def main(project=None):
    ok = True

    def item(name, fn):
        nonlocal ok
        t = time.time()
        try:
            r = fn()
            print("  OK    %-34s %s (%.1fs)" % (name, r or "", time.time() - t), flush=True)
        except Exception as e:
            ok = False
            print("  FAIL  %-34s %s" % (name, e), flush=True)
            traceback.print_exc()
    print("shapekit doctor  (frozen=%s, TORCH_HOME=%s)" % (getattr(sys, "frozen", False), os.environ.get("TORCH_HOME")), flush=True)

    def torch_cuda():
        import torch
        s = "torch %s" % torch.__version__
        if torch.cuda.is_available():
            x = torch.randn(512, 512, device="cuda"); (x @ x).sum().item()
            s += ", CUDA %s, %s" % (torch.version.cuda, torch.cuda.get_device_name(0))
        else:
            s += ", no CUDA (CPU mode)"
        return s
    item("PyTorch / GPU", torch_cuda)

    def feats():
        import numpy as np
        from ..learned import get
        lf = get("aliked", 1024, None)
        img = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype("uint8")
        f = lf.extract(img)
        m, _ = lf.match((f["xy"], f["desc"]), (f["xy"], f["desc"]), (640, 480), (640, 480))
        return "ALIKED %d keypoints, LightGlue %d self-matches" % (len(f["xy"]), len(m))
    item("ALIKED + LightGlue", feats)

    def retr():
        import numpy as np
        from ..learned.retrieval import GlobalDescriptor
        g = GlobalDescriptor("salad")
        d = g.describe([(np.random.default_rng(1).random((322, 434, 3)) * 255).astype("uint8")])
        return "DINOv2 + SALAD descriptor %d-D" % d.shape[1]
    item("DINOv2 / SALAD retrieval", retr)

    def seg():
        import numpy as np
        from ..masks import _Segmenter
        s = _Segmenter()
        lab = s.classes((np.random.default_rng(2).random((360, 480, 3)) * 255).astype("uint8"))
        return "DeepLabV3 ok, output %s" % (lab.shape,)
    item("AI masks network", seg)

    def ff():
        from ..video import ffmpeg_bin
        import subprocess
        out = subprocess.run([ffmpeg_bin(), "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        return out[:60]
    item("FFmpeg", ff)

    def pm():
        import numpy as np
        from ..learned.synthetic import synthetic_views, _shrink
        import tempfile, cv2
        p = os.path.join(tempfile.gettempdir(), "shapekit_doctor_tex.jpg")
        cv2.imwrite(p, (np.random.default_rng(3).random((512, 512, 3)) * 255).astype("uint8"))
        v = [_shrink(x, 0.5) for x in synthetic_views(p)]
        from ..learned.patchmatch import patchmatch_depth
        gt = v[2].gt
        d, _ = patchmatch_depth(v[2], [v[j] for j in (0, 1, 3, 4, 5)], 0.75 * gt[gt > 0].min(), 1.3 * gt[gt > 0].max())
        m = (d > 0) & (gt > 0)
        return "PatchMatch median depth error %.2f%%" % (100 * np.median(np.abs(d[m] - gt[m]) / gt[m]))
    item("PatchMatch depth (synthetic)", pm)

    def gui():
        os.environ.setdefault("QT_API", "pyside6")
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from ..gui.app import MainWindow
        w = MainWindow(project)
        w.show()
        for _ in range(20):
            app.processEvents(); time.sleep(0.05)
        w.model.plotter.render()
        n = len(w.chunk.cameras) if w.chunk else 0
        w.close()
        return "main window + 3D view ok" + (", project with %d cameras" % n if project else "")
    item("GUI (Qt + VTK)", gui)
    print("DOCTOR %s" % ("OK" if ok else "FAILED"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
