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
        s = "ALIKED %d keypoints, LightGlue %d self-matches" % (len(f["xy"]), len(m))
        mm = lf.memory_model()
        if mm:
            s += "; GPU memory %.0f MB + %.2f KB/pixel -> up to %.1f MP at native size right now" % (
                mm[0] / 1e6, mm[1] / 1e3, lf.max_pixels() / 1e6)
        return s
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

    def ba():
        import numpy as np
        from .. import camera as cm
        from ..learned.ba_gpu import available, bundle_adjust_gpu
        from ..ba import bundle_adjust
        rng = np.random.default_rng(0)
        N, M = 12, 600
        P = rng.uniform(-3, 3, (M, 3)); P[:, 2] = rng.uniform(-0.5, 0.5, M)
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        cr, ct = np.zeros((N, 3)), np.zeros((N, 3))
        for i in range(N):
            C = np.array([6 * np.cos(ang[i]), 6 * np.sin(ang[i]), 3.0])
            z = -C / np.linalg.norm(C); x = np.cross([0, 0, 1], z); x /= np.linalg.norm(x); y = np.cross(z, x)
            R = np.stack([x, y, z]); cr[i] = cm.R_to_rodrigues(R); ct[i] = -R @ C
        I = np.array([[1200, 960, 540, 0, 0, 0, 0, 0]], float)
        oc = np.repeat(np.arange(N), M); op = np.tile(np.arange(M), N)
        uv, zz = cm.project(cr[oc], ct[oc], P[op], I[np.zeros(len(oc), int)])
        uv = uv + rng.normal(0, 0.3, uv.shape)
        fixed = np.zeros(N, bool); fixed[0] = True
        args = (cr + rng.normal(0, 0.002, cr.shape), ct + rng.normal(0, 0.01, ct.shape), P + rng.normal(0, 0.01, P.shape),
                I, np.zeros(N, int), oc, op, uv)
        out = []
        for prec in (("double", "mixed") if available() else ()):
            r = bundle_adjust_gpu(*args, fixed_cams=fixed, iters=8, precision=prec)[4]["final_rms"]
            out.append("GPU %s rms %.3f" % (prec, r))
        r = bundle_adjust(*args, fixed_cams=fixed, iters=8)[4]["final_rms"]
        out.append("CPU rms %.3f" % r)
        return ", ".join(out)
    item("Bundle adjustment (GPU + CPU)", ba)

    def depth_gpu():
        import numpy as np, tempfile, cv2
        from ..learned.synthetic import synthetic_views, _shrink
        from ..learned.mvs_gpu import gpu_available, compute_depth_gpu
        from .. import raster
        p = os.path.join(tempfile.gettempdir(), "shapekit_doctor_tex.jpg")
        cv2.imwrite(p, (np.random.default_rng(3).random((512, 512, 3)) * 255).astype("uint8"))
        v = [_shrink(x, 0.5) for x in synthetic_views(p)]
        gt = v[2].gt
        s = []
        if gpu_available():
            d, _ = compute_depth_gpu(v[2], [v[j] for j in (0, 1, 3, 4, 5)], 0.75 * gt[gt > 0].min(), 1.3 * gt[gt > 0].max(), 128, 7)
            m = (d > 0) & (gt > 0)
            s.append("plane sweep median error %.2f%%" % (100 * np.median(np.abs(d[m] - gt[m]) / gt[m])))
        g = np.random.default_rng(4)
        c = g.random((10000, 1, 3)) * [2, 2, 1] - [1, 1, 0] + [0, 0, 3]             # small triangles
        V = (c + g.normal(0, 0.01, (10000, 3, 3))).reshape(-1, 3); F = np.arange(30000).reshape(-1, 3)
        uv, z = raster.project_vertices(V, np.zeros(3), np.zeros(3), np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1.0]]))
        ids, _ = raster.rasterize(uv, z, F, 640, 480)
        s.append("rasteriser %d px covered" % int((ids >= 0).sum()))
        return ", ".join(s)
    item("Plane-sweep depth + rasteriser", depth_gpu)

    def ff():
        from ..ffmpeg import ffmpeg_bin, version
        ff = ffmpeg_bin(); ffmpeg_bin("ffprobe")
        return "%s  [%s]" % (version(ff)[:48], ff)
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
