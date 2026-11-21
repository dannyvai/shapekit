"""3D Model view (PyVista/VTK): tie points, dense cloud, mesh (shaded / coloured / textured), cameras,
markers, region box editing, rectangle + free-form selection, point picking and measurement."""
import os
import numpy as np

os.environ.setdefault("QT_API", "pyside6")
import pyvista as pv                                   # noqa: E402
import vtk                                             # noqa: E402
from pyvistaqt import QtInteractor                     # noqa: E402
from PySide6.QtCore import Qt, QEvent, QRect, QPoint, QSize, Signal   # noqa: E402
from PySide6.QtWidgets import QWidget, QVBoxLayout       # noqa: E402

from .. import edit                                    # noqa: E402
from ..io import read_ply                              # noqa: E402

DISPLAY_MODES = ["Tie points", "Point cloud", "Mesh (shaded)", "Mesh (colours)", "Mesh (textured)"]


class ModelView(QWidget):
    picked = Signal(object)            # np.array(3) world(internal) point, emitted in pick/measure modes
    selection_changed = Signal(str, int)
    region_changed = Signal(object)    # region dict
    camera_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        lay.addWidget(self.plotter.interactor)
        self.plotter.set_background("#30343c", top="#59606e")
        self.plotter.enable_trackball_style()
        self.iw = self.plotter.interactor
        self.mode = "navigate"          # navigate | rect | lasso | pick | measure
        self.select_mode = "replace"
        self.target = "mesh"
        self.display = "Mesh (textured)"
        self.show_cameras = True
        self.show_region = True
        self.show_markers = True
        self.chunk = None
        self.data = {}
        self.sel = {}
        self._press = None
        self._lasso = []
        self._lasso_actor = None
        self._box = None
        self.measure_pts = []
        self.iw.installEventFilter(self)

    # ------------------------------------------------------------------ data
    def set_chunk(self, chunk, reset_camera=False):
        self.chunk = chunk
        self.data, self.sel = {}, {}
        self.reload(reset_camera=reset_camera)

    def _load(self, layer):
        ch = self.chunk
        if ch is None:
            return None
        if layer in self.data:
            return self.data[layer]
        d = None
        try:
            if layer == "tie":
                sp = ch.load_sparse()
                if sp is not None:
                    d = dict(P=sp["points"], C=sp["colors"])
            elif layer == "dense":
                p = os.path.join(ch.dir, "point_cloud.ply")
                if os.path.exists(p):
                    r = read_ply(p)
                    d = dict(P=r["verts"], C=r.get("colors"))
            elif layer == "mesh":
                p = os.path.join(ch.dir, "mesh.ply")
                if os.path.exists(p):
                    r = read_ply(p)
                    d = dict(V=r["verts"], F=r["faces"], C=r.get("colors"))
                    d["cen"] = r["verts"][r["faces"]].mean(1)
                    uvp = os.path.join(ch.dir, "model", "uv.npz")
                    tex = os.path.join(ch.dir, "model", "model_texture.jpg")
                    if os.path.exists(uvp) and os.path.exists(tex):
                        u = np.load(uvp)
                        if len(u["face_uv"]) == len(d["F"]):
                            d["uv"], d["face_uv"], d["tex"] = u["uv"], u["face_uv"], tex
        except Exception as e:
            print("load", layer, e)
        self.data[layer] = d
        return d

    def invalidate(self, *layers):
        for l in layers or list(self.data):
            self.data.pop(l, None); self.sel.pop(l, None)

    # ------------------------------------------------------------------ drawing
    def reload(self, reset_camera=False):
        pl = self.plotter
        for name in ("tie", "dense", "mesh", "sel", "cams", "cam_sel", "region", "markers", "measure"):
            pl.remove_actor(name, render=False)
        ch = self.chunk
        if ch is None:
            pl.render(); return
        mode = self.display
        if mode == "Tie points":
            d = self._load("tie")
            if d:
                pl.add_points(pv.PolyData(np.asarray(d["P"], np.float32)), scalars=d["C"], rgb=True, point_size=3,
                              name="tie", reset_camera=False)
            self.target = "tie"
        elif mode == "Point cloud":
            d = self._load("dense")
            if d:
                step = max(1, len(d["P"]) // 4_000_000)
                pts = pv.PolyData(np.asarray(d["P"][::step], np.float32))
                kw = dict(scalars=d["C"][::step], rgb=True) if d.get("C") is not None else dict(color="#ddd")
                pl.add_points(pts, point_size=2, name="dense", reset_camera=False, **kw)
            self.target = "dense"
        else:
            d = self._load("mesh")
            if d:
                faces = np.hstack([np.full((len(d["F"]), 1), 3), d["F"]]).ravel()
                if mode == "Mesh (textured)" and "uv" in d:
                    cu = d["uv"][d["face_uv"].ravel()]
                    q = np.round(cu * 2 ** 20).astype(np.int64)
                    key = np.stack([d["F"].ravel().astype(np.int64), q[:, 0], q[:, 1]], 1)
                    uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
                    m = pv.PolyData(d["V"][uniq[:, 0]].astype(np.float32),
                                    np.hstack([np.full((len(d["F"]), 1), 3), inv.reshape(-1, 3)]).ravel())
                    tc = cu[first].astype(np.float32)
                    try:
                        m.active_texture_coordinates = tc
                    except AttributeError:
                        m.t_coords = tc
                    pl.add_mesh(m, texture=pv.read_texture(d["tex"]), name="mesh", lighting=False,
                                reset_camera=False)
                else:
                    m = pv.PolyData(d["V"].astype(np.float32), faces)
                    if mode == "Mesh (colours)" and d.get("C") is not None:
                        m["rgb"] = d["C"]
                        pl.add_mesh(m, scalars="rgb", rgb=True, name="mesh", lighting=False, reset_camera=False)
                    else:
                        pl.add_mesh(m, color="#d9d4c7", name="mesh", smooth_shading=False, specular=0.15,
                                    reset_camera=False)
            self.target = "mesh"
        self._draw_selection()
        if self.show_cameras:
            self._draw_cameras()
        if self.show_region and ch.region and self._box is None:
            self._draw_region()
        if self.show_markers:
            self._draw_markers()
        if reset_camera:
            self.reset_view()
        pl.render()

    def scene_extent(self):
        if self.chunk is not None and self.chunk.region:
            return float(np.linalg.norm(self.chunk.region["size"]))
        return 1.0

    def _draw_cameras(self, highlight=None):
        ch = self.chunk
        al = ch.aligned_cameras()
        if not al:
            return
        from .. import camera as cm
        s = self.scene_extent() * 0.02
        pts, lines = [], []
        for i in al:
            c = ch.cameras[i]; sen = ch.sensors[c.sensor]
            R = cm.rodrigues_to_R(c.rvec)[0]; C = -R.T @ c.tvec
            f = sen.intr[0]; hw, hh = sen.width / 2 / f, sen.height / 2 / f
            corners = [C + s * (R[2] + sx * hw * R[0] + sy * hh * R[1]) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            b = len(pts)
            pts += [C] + corners
            for k in range(4):
                lines += [2, b, b + 1 + k, 2, b + 1 + k, b + 1 + (k + 1) % 4]
        poly = pv.PolyData(np.array(pts, np.float32), lines=np.array(lines))
        self.plotter.add_mesh(poly, color="#4aa3ff", line_width=1, name="cams", reset_camera=False, pickable=False)
        if highlight is not None and ch.cameras[highlight].aligned:
            k = al.index(highlight)
            sub = poly.extract_cells(range(k * 8, k * 8 + 8))
            self.plotter.add_mesh(sub, color="#ffcc33", line_width=3, name="cam_sel", reset_camera=False, pickable=False)

    def highlight_camera(self, i):
        self.plotter.remove_actor("cam_sel", render=False)
        if self.show_cameras and self.chunk is not None:
            self._draw_cameras(highlight=i)
        self.plotter.render()

    def _draw_region(self):
        cor = edit.region_corners(self.chunk.region)
        idx = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        lines = np.array([[2, a, b] for a, b in idx]).ravel()
        self.plotter.add_mesh(pv.PolyData(cor.astype(np.float32), lines=lines), color="#ff9f1a", line_width=2,
                              name="region", reset_camera=False, pickable=False)

    def _draw_markers(self):
        ms = [m for m in (self.chunk.markers or []) if m.get("position") is not None]
        if not ms:
            return
        P = np.array([m["position"] for m in ms], np.float32)
        kw = dict(point_size=12, font_size=13, point_color="#3dff6e", text_color="white", shape_opacity=0.35,
                  always_visible=True, name="markers", render_points_as_spheres=True)
        try:
            self.plotter.add_point_labels(P, [m["label"] for m in ms], reset_camera=False, **kw)
        except TypeError:
            self.plotter.add_point_labels(P, [m["label"] for m in ms], **kw)

    def reset_view(self):
        ch = self.chunk
        if ch is None or not ch.region:
            self.plotter.reset_camera(); return
        c = np.array(ch.region["center"]); R = np.array(ch.region["R"]); e = self.scene_extent()
        self.plotter.camera.focal_point = tuple(c)
        self.plotter.camera.position = tuple(c - R[1] * 0.9 * e + R[2] * 0.7 * e)
        self.plotter.camera.up = tuple(R[2])
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def view_from(self, axis):
        ch = self.chunk
        if ch is None or not ch.region:
            return
        c = np.array(ch.region["center"]); R = np.array(ch.region["R"]); e = self.scene_extent()
        if axis == "top":
            pos, up = c + R[2] * 1.3 * e, R[1]
        elif axis == "front":
            pos, up = c - R[1] * 1.3 * e, R[2]
        else:
            pos, up = c + R[0] * 1.3 * e, R[2]
        self.plotter.camera.focal_point = tuple(c); self.plotter.camera.position = tuple(pos)
        self.plotter.camera.up = tuple(up); self.plotter.reset_camera_clipping_range(); self.plotter.render()

    # ------------------------------------------------------------------ selection
    def _target_points(self):
        if self.target == "mesh":
            d = self._load("mesh")
            return None if d is None else d["cen"]
        d = self._load(self.target)
        return None if d is None else d["P"]

    def _composite_matrix(self):
        w, h = self.iw.width(), self.iw.height()
        cam = self.plotter.renderer.GetActiveCamera()
        m = cam.GetCompositeProjectionTransformMatrix(w / max(h, 1), -1, 1)
        return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)]), w, h

    def select_polygon(self, poly):
        X = self._target_points()
        if X is None:
            return
        M, w, h = self._composite_matrix()
        new = edit.select_in_polygon(X, M, w, h, poly)
        self.sel[self.target] = edit.combine(self.sel.get(self.target), new, self.select_mode)
        self._draw_selection(); self.plotter.render()
        self.selection_changed.emit(self.target, int(self.sel[self.target].sum()))

    def set_selection(self, layer, mask):
        self.sel[layer] = mask
        self._draw_selection(); self.plotter.render()
        self.selection_changed.emit(layer, int(mask.sum()) if mask is not None else 0)

    def clear_selection(self):
        self.sel = {}
        self.plotter.remove_actor("sel", render=False)
        self.plotter.render()
        self.selection_changed.emit(self.target, 0)

    def _draw_selection(self):
        self.plotter.remove_actor("sel", render=False)
        m = self.sel.get(self.target)
        if m is None or not m.any():
            return
        if self.target == "mesh":
            d = self._load("mesh")
            F = d["F"][m]
            used = np.unique(F); remap = -np.ones(len(d["V"]), int); remap[used] = np.arange(len(used))
            poly = pv.PolyData(d["V"][used].astype(np.float32),
                               np.hstack([np.full((len(F), 1), 3), remap[F]]).ravel())
            self.plotter.add_mesh(poly, color="#ff3355", opacity=0.75, name="sel", reset_camera=False, pickable=False)
        else:
            d = self._load(self.target)
            P = d["P"][m]
            if len(P) > 2_000_000:
                P = P[:: len(P) // 2_000_000]
            self.plotter.add_points(pv.PolyData(np.asarray(P, np.float32)), color="#ff3355", point_size=4,
                                    name="sel", reset_camera=False, pickable=False)

    def selection(self):
        return self.target, self.sel.get(self.target)

    # ------------------------------------------------------------------ region editing
    def edit_region(self, on):
        """Box widget placed on the unit cube, then transformed onto the region; the widget's own
        transform (unit cube -> current box) is read back when editing ends."""
        pl = self.plotter
        if on and self.chunk is not None and self.chunk.region:
            pl.remove_actor("region", render=False)
            self._box = pl.add_box_widget(lambda *a: None, bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), factor=1.0,
                                          rotation_enabled=True, pass_widget=True, color="#ff9f1a",
                                          outline_translation=True)
            t = vtk.vtkTransform(); t.SetMatrix(edit.region_matrix(self.chunk.region).ravel().tolist())
            self._box.SetTransform(t)
            pl.render()
        elif not on and self._box is not None:
            t = vtk.vtkTransform(); self._box.GetTransform(t)
            mm = t.GetMatrix()
            M = np.array([[mm.GetElement(i, j) for j in range(4)] for i in range(4)])
            pl.clear_box_widgets()
            self._box = None
            reg = edit.region_from_matrix(M)
            if self.chunk is not None and min(reg["size"]) > 1e-9:
                self.chunk.region = reg
                self.region_changed.emit(reg)
            if self.show_region and self.chunk is not None:
                self._draw_region()
            pl.render()

    # ------------------------------------------------------------------ picking / measuring
    def pick_at(self, x, y):
        dpr = self.iw.devicePixelRatioF()
        h = self.iw.height()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.003)
        if picker.Pick(x * dpr, (h - y) * dpr, 0, self.plotter.renderer) and picker.GetCellId() >= 0:
            return np.array(picker.GetPickPosition())
        return None

    def draw_measure(self, pts):
        self.plotter.remove_actor("measure", render=False)
        if len(pts) >= 1:
            P = np.array(pts, np.float32)
            poly = pv.PolyData(P, lines=np.array([2, 0, 1]) if len(P) == 2 else None)
            self.plotter.add_mesh(poly, color="#ffe066", line_width=3, point_size=10, render_points_as_spheres=True,
                                  name="measure", reset_camera=False, pickable=False)
        self.plotter.render()

    # ------------------------------------------------------------------ mouse handling
    def set_mode(self, mode):
        self.mode = mode
        self.iw.setCursor(Qt.CrossCursor if mode != "navigate" else Qt.ArrowCursor)

    def _lasso_draw(self):
        ren = self.plotter.renderer
        if self._lasso_actor is not None:
            ren.RemoveViewProp(self._lasso_actor); self._lasso_actor = None
        if len(self._lasso) < 2:
            self.plotter.render(); return
        dpr = self.iw.devicePixelRatioF(); h = self.iw.height()
        pts = vtk.vtkPoints()
        for x, y in self._lasso:
            pts.InsertNextPoint(x * dpr, (h - y) * dpr, 0)
        line = vtk.vtkCellArray(); line.InsertNextCell(len(self._lasso) + 1)
        for k in range(len(self._lasso)):
            line.InsertCellPoint(k)
        line.InsertCellPoint(0)
        pd = vtk.vtkPolyData(); pd.SetPoints(pts); pd.SetLines(line)
        mp = vtk.vtkPolyDataMapper2D(); mp.SetInputData(pd)
        act = vtk.vtkActor2D(); act.SetMapper(mp)
        act.GetProperty().SetColor(1, 0.85, 0.2); act.GetProperty().SetLineWidth(2)
        ren.AddViewProp(act); self._lasso_actor = act
        self.plotter.render()

    def eventFilter(self, obj, ev):
        if obj is not getattr(self, "iw", None) or getattr(self, "mode", "navigate") == "navigate":
            return False
        t = ev.type()
        if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
            p = ev.position().toPoint()
            mods = ev.modifiers()
            self.select_mode = "add" if mods & Qt.ShiftModifier else ("subtract" if mods & Qt.ControlModifier else "replace")
            if self.mode == "rect":
                self._press = p
            elif self.mode == "lasso":
                self._lasso = [(p.x(), p.y())]
            elif self.mode in ("pick", "measure"):
                X = self.pick_at(p.x(), p.y())
                if X is not None:
                    self.picked.emit(X)
            return True
        if t == QEvent.MouseMove and ev.buttons() & Qt.LeftButton:
            p = ev.position().toPoint()
            if self.mode == "rect" and self._press is not None:
                a = self._press
                self._lasso = [(a.x(), a.y()), (p.x(), a.y()), (p.x(), p.y()), (a.x(), p.y())]
                self._lasso_draw()
            elif self.mode == "lasso":
                if not self._lasso or abs(self._lasso[-1][0] - p.x()) + abs(self._lasso[-1][1] - p.y()) > 3:
                    self._lasso.append((p.x(), p.y()))
                    self._lasso_draw()
            return True
        if t == QEvent.MouseButtonRelease and ev.button() == Qt.LeftButton:
            if self.mode == "rect" and self._press is not None:
                poly = self._lasso; self._lasso = []; self._lasso_draw(); self._press = None
                xs = [q[0] for q in poly]; ys = [q[1] for q in poly]
                if poly and max(xs) - min(xs) > 2 and max(ys) - min(ys) > 2:
                    self.select_polygon(poly)
                elif self.select_mode == "replace":
                    self.clear_selection()
            elif self.mode == "lasso":
                poly = self._lasso; self._lasso = []; self._lasso_draw()
                if len(poly) > 2:
                    self.select_polygon(poly)
            return True
        return False

    def close(self):
        try:
            self.plotter.close()
        except Exception:
            pass
