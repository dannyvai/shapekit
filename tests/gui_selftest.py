"""Automated GUI smoke test: drives the real MainWindow on a COPY of a project and saves screenshots.

uv run python -m tests.gui_selftest projects\\gpu_test
Outputs: <project>_guitest/selftest/*.png and selftest.log
"""
import os
import shutil
import sys
import time
import traceback

import numpy as np


def main(src):
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6.QtWidgets import QApplication, QMessageBox, QInputDialog
    from PySide6.QtCore import Qt
    dst = os.path.abspath(src.rstrip("\\/") + "_guitest")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("depth", "features", "exports", "*.html"))
    out = os.path.join(dst, "selftest"); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "selftest.log"), "w", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    # never block on dialogs
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: log("  [info]", a[2] if len(a) > 2 else ""))
    QMessageBox.critical = staticmethod(lambda *a, **k: log("  [CRITICAL]", a[2] if len(a) > 2 else ""))
    QInputDialog.getText = staticmethod(lambda *a, **k: ("selftest marker", True))
    QInputDialog.getDouble = staticmethod(lambda *a, **k: (2.0, True))

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    from shapekit.gui.app import MainWindow
    w = MainWindow(dst)
    w.resize(1600, 1000); w.show()

    def pump(sec=0.5):
        t = time.time()
        while time.time() - t < sec:
            app.processEvents(); time.sleep(0.02)

    def shot(name):
        pump(0.6)
        try:
            w.model.plotter.render()
            w.model.plotter.screenshot(os.path.join(out, name + "_3d.png"))
        except Exception as e:
            log("  3d screenshot failed", e)
        w.grab().save(os.path.join(out, name + "_window.png"))

    ok, fail = 0, 0

    def step(name, fn):
        nonlocal ok, fail
        log("STEP", name)
        try:
            r = fn()
            pump(0.3)
            log("  ok", "" if r is None else r, "| status:", w.status_lbl.text())
            ok += 1
        except Exception:
            log("  FAILED\n" + traceback.format_exc()); fail += 1

    errors = []
    _hook = sys.excepthook
    sys.excepthook = lambda *e: (errors.append("".join(traceback.format_exception(*e))), _hook(*e))
    pump(2)
    ch = w.chunk
    log("project", dst, "cameras", len(ch.cameras), "markers", len(ch.markers))
    step("initial view", lambda: shot("01_initial"))
    for k, mode in enumerate(["Tie points", "Point cloud", "Mesh (shaded)", "Mesh (colours)", "Mesh (textured)"]):
        step("display " + mode, lambda m=mode, k=k: (w.set_display(m), shot("02_display_%d" % k)))
    step("top view", lambda: (w.model.view_from("top"), shot("03_top")))
    step("reset view", lambda: (w.model.reset_view(), shot("03_reset")))

    def rect_select():
        w.set_display("Mesh (shaded)"); w.set_tool("rect")
        W, H = w.model.iw.width(), w.model.iw.height()
        w.model.select_mode = "replace"
        w.model.select_polygon([(W * .35, H * .35), (W * .65, H * .35), (W * .65, H * .65), (W * .35, H * .65)])
        n = int(w.model.selection()[1].sum())
        shot("04_rect_selection")
        assert n > 0, "nothing selected"
        return "%d faces selected" % n
    step("rect select faces", rect_select)

    def lasso_select():
        W, H = w.model.iw.width(), w.model.iw.height()
        w.model.select_mode = "add"
        poly = [(W * .5 + W * .12 * np.cos(t), H * .25 + H * .1 * np.sin(t)) for t in np.linspace(0, 6.28, 24)]
        w.model.select_polygon(poly)
        return "%d faces selected" % int(w.model.selection()[1].sum())
    step("free-form add", lasso_select)

    def mouse_drag(points, mods=None):
        """Real Qt mouse events on the 3D widget (exercises the outline drawing + event filter)."""
        from PySide6.QtCore import QPointF, QEvent
        from PySide6.QtGui import QMouseEvent
        mods = mods or Qt.NoModifier
        iw = w.model.iw

        def send(kind, p, buttons):
            pt = QPointF(*p)
            ev = QMouseEvent(kind, pt, iw.mapToGlobal(pt), Qt.LeftButton, buttons, mods)
            QApplication.sendEvent(iw, ev)
            app.processEvents()
        send(QEvent.MouseButtonPress, points[0], Qt.LeftButton)
        for p in points[1:]:
            send(QEvent.MouseMove, p, Qt.LeftButton)
        send(QEvent.MouseButtonRelease, points[-1], Qt.NoButton)

    def rect_mouse():
        w.model.clear_selection(); w.set_display("Mesh (shaded)"); w.set_tool("rect")
        W, H = w.model.iw.width(), w.model.iw.height()
        mouse_drag([(W * .3, H * .3), (W * .5, H * .5), (W * .7, H * .7)])
        n = int(w.model.selection()[1].sum()) if w.model.selection()[1] is not None else 0
        assert n > 0, "rect drag selected nothing"
        shot("04b_rect_mouse")
        return "%d faces via mouse drag" % n
    step("rect select with real mouse events", rect_mouse)

    def lasso_mouse():
        w.model.clear_selection(); w.set_tool("lasso")
        W, H = w.model.iw.width(), w.model.iw.height()
        pts = [(W * .5 + W * .15 * np.cos(t), H * .5 + H * .15 * np.sin(t)) for t in np.linspace(0, 6.28, 40)]
        mouse_drag(pts)
        n = int(w.model.selection()[1].sum()) if w.model.selection()[1] is not None else 0
        assert n > 0, "lasso drag selected nothing"
        shot("04c_lasso_mouse")
        w.set_tool("navigate")
        return "%d faces via free-form drag" % n
    step("free-form select with real mouse events", lasso_mouse)

    def delete():
        nf = ch.stats["mesh"]["faces"]
        w.delete_selection()
        pump(1)
        w.set_display("Mesh (textured)"); shot("05_after_delete_textured")
        return "faces %d -> %d" % (nf, w.chunk.stats["mesh"]["faces"])
    step("delete selection (textured mesh stays valid)", delete)
    step("clear selection", lambda: (w.model.clear_selection(), w.set_tool("navigate")))

    def region():
        before = np.array(w.chunk.region["size"])
        w.a_region_edit.setChecked(True); w.toggle_region_edit(); shot("06_region_edit")
        w.a_region_edit.setChecked(False); w.toggle_region_edit()
        after = np.array(w.chunk.region["size"])
        assert np.allclose(before, after, rtol=1e-4), "region changed %s -> %s" % (before, after)
        # programmatic edit: shrink the box via the widget and check it is read back
        w.a_region_edit.setChecked(True); w.toggle_region_edit()
        import vtk
        t = vtk.vtkTransform(); w.model._box.GetTransform(t); t.PostMultiply(); 
        c = np.array(w.chunk.region["center"])
        t.Translate(*(-c)); t.Scale(0.8, 0.8, 0.8); t.Translate(*c)
        w.model._box.SetTransform(t)
        w.a_region_edit.setChecked(False); w.toggle_region_edit()
        after2 = np.array(w.chunk.region["size"])
        assert np.allclose(after2, before * 0.8, rtol=1e-3), "scaled region %s" % after2
        shot("06b_region_scaled")
        return "region size kept %s, scaled -> %s" % (np.round(before, 2).tolist(), np.round(after2, 2).tolist())
    step("region edit on/off", region)

    def photo():
        m = max(ch.markers, key=lambda m: len(m["projections"]))
        cam = int(sorted(m["projections"], key=lambda k: m["projections"][k].get("error_px", 9))[0])
        w.open_photo(cam); pump(1)
        pv = w.tabs.currentWidget(); pv.cb_tie.setChecked(True); pump(0.5)
        # zoom on marker
        xy = m["projections"][str(cam)]["xy"]
        pv.view.resetTransform(); pv.view.scale(4, 4); pv.view.centerOn(xy[0], xy[1]); pump(0.5)
        w.grab().save(os.path.join(out, "07_photo_view_window.png"))
        w.tabs.setCurrentIndex(0)
        return "camera %s, marker %s" % (ch.cameras[cam].label, m["label"])
    step("photo view with markers", photo)

    def place_marker():
        cams = ch.aligned_cameras()
        sen = ch.sensors[0]
        w._place("", cams[5], sen.width / 2, sen.height / 2)
        w._place("selftest marker", cams[8], sen.width / 2 + 10, sen.height / 2)
        from shapekit import markers as MK
        m = MK.find_marker(w.chunk, "selftest marker")
        return "projections %d, position %s" % (len(m["projections"]), m["position"] is not None)
    step("place marker in photos (right-click workflow)", place_marker)

    def scalebar():
        labs = [m["label"] for m in ch.markers if m.get("position") is not None and m["label"].startswith("target")][:2]
        from PySide6.QtCore import QItemSelectionModel
        t = w.reference.mt
        t.clearSelection()
        for r in range(t.rowCount()):
            if t.item(r, 0).data(Qt.UserRole) in labs:
                t.selectionModel().select(t.model().index(r, 0),
                                          QItemSelectionModel.Select | QItemSelectionModel.Rows)
        sel = w.reference.selected_markers()
        assert len(sel) == 2, "selected %s" % sel
        w._reference_request("add_scalebar")
        pump(0.5); w.ref_dock.raise_()
        w.grab().save(os.path.join(out, "08_reference_window.png"))
        return "scale bar %s, transform %s, scale %.5f" % (sel, w.chunk.transform is not None,
                                                         w.chunk.stats["reference"]["scale"])
    step("scale bar + update", scalebar)

    def measure():
        w.set_display("Mesh (shaded)"); w.set_tool("measure")
        W, H = w.model.iw.width(), w.model.iw.height()
        pts = []
        for x, y in [(W * .4, H * .5), (W * .6, H * .5)]:
            X = w.model.pick_at(x, y)
            if X is not None:
                w.model.picked.emit(X); pts.append(X)
        shot("09_measure")
        w.set_tool("navigate")
        return "picked %d points" % len(pts)
    step("ruler", measure)

    def pick_marker():
        w.set_tool("pick")
        W, H = w.model.iw.width(), w.model.iw.height()
        X = None
        for fx in (.3, .7, .25, .75):
            X = w.model.pick_at(W * fx, H * .6)
            if X is not None:
                break
        if X is not None:
            w.model.picked.emit(X)
        w.set_tool("navigate"); shot("10_markers_on_model")
        return "picked %s" % (X is not None)
    step("add marker on model", pick_marker)

    def gradual():
        from shapekit import edit
        e = edit.tie_point_errors(w.chunk)
        w.set_display("Tie points"); w.model.set_selection("tie", e > 1.0)
        shot("11_gradual_selection")
        return "%d tie points above 1 px" % (e > 1.0).sum()
    step("gradual selection", gradual)

    def photos_filter():
        i = w.photos.filter.findText("With marker: target 34")
        if i >= 0:
            w.photos.filter.setCurrentIndex(i)
        pump(2)
        w.grab().save(os.path.join(out, "12_photos_filter_window.png"))
        return "visible photos %d" % sum(not w.photos.list.item(k).isHidden() for k in range(w.photos.list.count()))
    step("photos filtered by marker", photos_filter)

    # ---------------------------------------------------------------- roadmap features
    from shapekit.gui import common as _common
    answers = {}

    def fake_ask(title, fields, parent=None, note=None):
        vals = {k: d for k, _l, _kind, d, _o in fields}
        vals.update(answers.get(title, {}))
        log("  [dialog] %s -> %s" % (title, {k: v for k, v in vals.items() if not str(k).startswith(("model", "f", "fixed"))}))
        return vals
    _common.ParamDialog.ask = staticmethod(fake_ask)
    from shapekit.gui import app as _app
    _app.ParamDialog.ask = staticmethod(fake_ask)

    def wait_task(limit=1800):
        t0 = time.time()
        while w.task is not None and time.time() - t0 < limit:
            pump(0.2)
        assert w.task is None, "task still running"

    def mask_paint():
        from PySide6.QtCore import QPointF, QEvent
        from PySide6.QtGui import QMouseEvent
        from shapekit import masks as MS
        i = w.chunk.aligned_cameras()[2]
        w.open_photo(i); pump(1)
        pv = w.tabs.currentWidget()
        pv.cb_mask.setChecked(True); pv.view.fit(); pump(0.3)
        vp = pv.view.viewport()
        W, H = vp.width(), vp.height()

        def send(kind, p, btn, buttons, mods=Qt.NoModifier):
            pt = QPointF(*p)
            QApplication.sendEvent(vp, QMouseEvent(kind, pt, vp.mapToGlobal(pt), btn, buttons, mods)); app.processEvents()
        pts = [(W * .3 + k * W * .02, H * .5) for k in range(15)]
        send(QEvent.MouseButtonPress, pts[0], Qt.LeftButton, Qt.LeftButton)
        for p in pts[1:]:
            send(QEvent.MouseMove, p, Qt.NoButton, Qt.LeftButton)
        send(QEvent.MouseButtonRelease, pts[-1], Qt.LeftButton, Qt.NoButton)
        m1 = MS.load_mask(w.chunk, i)
        assert m1 is not None and (~m1).sum() > 100, "brush stroke did not mask anything"
        poly = [(W * .6 + W * .08 * np.cos(a), H * .3 + H * .1 * np.sin(a)) for a in np.linspace(0, 6.28, 20)]
        send(QEvent.MouseButtonPress, poly[0], Qt.LeftButton, Qt.LeftButton, Qt.ControlModifier)
        for p in poly[1:]:
            send(QEvent.MouseMove, p, Qt.NoButton, Qt.LeftButton, Qt.ControlModifier)
        send(QEvent.MouseButtonRelease, poly[-1], Qt.LeftButton, Qt.NoButton, Qt.ControlModifier)
        m2 = MS.load_mask(w.chunk, i)
        assert (~m2).sum() > (~m1).sum(), "free-form outline did not add to the mask"
        pump(0.5); w.grab().save(os.path.join(out, "13_mask_editing_window.png"))
        n = int((~m2).sum())
        pv.clear_mask(); assert MS.load_mask(w.chunk, i) is None
        pv.cb_mask.setChecked(False); w.tabs.setCurrentIndex(0)
        return "masked %d px with brush + outline, then cleared" % n
    step("mask painting in the photo view (real mouse events)", mask_paint)

    def calib():
        sen = w.chunk.sensors[0]
        before = (sen.model, float(sen.intr[0]))
        w.camera_calibration()
        assert (sen.model, float(sen.intr[0])) == before, "unchanged dialog changed the sensor"
        sen.set_model("fisheye_equidistant"); w.workspace.refresh(w.project, w.chunk); pump(0.3)
        sen.set_model("frame"); sen.intr[0] = before[1]; w.save()
        return "camera calibration dialog ok (%s, f %.1f)" % before
    step("camera calibration dialog", calib)

    def chunks():
        n0 = len(w.project.chunks)
        w.chunk_action("add", -1); pump(0.5)
        assert len(w.project.chunks) == n0 + 1 and w.chunk_idx == n0, "add chunk"
        w.set_chunk(0); pump(0.5)
        answers["Duplicate Chunk"] = dict(label="dup", depth=False, cloud=False, mesh=False)
        w.chunk_action("duplicate", 0); wait_task(); pump(0.5)
        dup = len(w.project.chunks) - 1
        answers["Align Chunks"] = dict(ref="0: %s" % w.project.chunks[0].label, by="cameras")
        w.chunk_action("align", -1); pump(0.3)
        assert w.project.chunks[dup].transform is not None, "align chunks did not set a transform"
        answers["Merge Chunks"] = {**{"c%d" % k: k in (0, dup) for k in range(len(w.project.chunks))},
                                   "label": "merged", "depth": False, "cloud": False, "mesh": False}
        w.chunk_action("merge", -1); wait_task(); pump(1)
        merged = w.chunk
        assert merged.label == "merged" and len(merged.cameras) == 2 * len(w.project.chunks[0].cameras), "merge"
        w.ws_dock.raise_(); shot("14_merged_chunk")
        for k in range(len(w.project.chunks) - 1, 0, -1):
            w.chunk_action("remove", k)
        w.set_chunk(0); pump(0.5)
        return "add / switch / duplicate / align / merge / remove ok; merged chunk had %d cameras" % len(merged.cameras)
    step("chunks: add, duplicate, align, merge, remove", chunks)

    def holes():
        nf = w.chunk.stats["mesh"]["faces"]
        answers["Close Holes"] = dict(n=200)
        w.close_holes(); wait_task(); pump(0.5)
        shot("15_closed_holes")
        return "faces %d -> %d" % (nf, w.chunk.stats["mesh"]["faces"])
    step("close holes (background task)", holes)

    if errors:
        fail += len(errors)
        log("\nUNCAUGHT EXCEPTIONS IN EVENT HANDLERS: %d\n%s" % (len(errors), errors[0]))
    log("\nRESULT: %d passed, %d failed" % (ok, fail))
    w.close(); pump(0.5)
    logf.close()
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "projects/gpu_test"))
