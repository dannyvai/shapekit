"""shapekit desktop GUI main window (Metashape-style layout)."""
import os
import sys
import types
import numpy as np

os.environ.setdefault("QT_API", "pyside6")
from PySide6.QtCore import Qt, QUrl, QSettings                         # noqa: E402
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QDesktopServices   # noqa: E402
from PySide6.QtWidgets import (QMainWindow, QApplication, QDockWidget, QTabWidget, QFileDialog, QMessageBox,  # noqa: E402
                               QToolBar, QComboBox, QLabel, QProgressBar, QInputDialog)

from ..project import Project                                          # noqa: E402
from .. import util, edit, markers as MK                               # noqa: E402
from .common import LogBridge, Task, ParamDialog                       # noqa: E402
from .model_view import ModelView, DISPLAY_MODES                       # noqa: E402
from .photo_view import PhotoView                                      # noqa: E402
from .panes import WorkspacePane, PhotosPane, ReferencePane, ConsolePane   # noqa: E402

FEATURES = ["auto", "aliked", "superpoint", "disk", "sift"]


class MainWindow(QMainWindow):
    def __init__(self, project_dir=None):
        super().__init__()
        self.setWindowTitle("shapekit")
        self.resize(1600, 980)
        self.project = None
        self.chunk = None
        self.chunk_idx = 0
        self.task = None
        self.settings = QSettings("shapekit", "shapekit")

        self.tabs = QTabWidget(); self.tabs.setTabsClosable(True); self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.model = ModelView()
        self.tabs.addTab(self.model, "Model")
        self.tabs.tabBar().setTabButton(0, self.tabs.tabBar().ButtonPosition.RightSide, None)
        self.setCentralWidget(self.tabs)

        self.workspace = WorkspacePane(); self.photos = PhotosPane(); self.reference = ReferencePane()
        self.console = ConsolePane()
        self.ws_dock = self._dock("Workspace", self.workspace, Qt.LeftDockWidgetArea)
        self.ref_dock = self._dock("Reference", self.reference, Qt.LeftDockWidgetArea)
        self.tabifyDockWidget(self.ws_dock, self.ref_dock)
        self.ws_dock.raise_()
        self.photos_dock = self._dock("Photos", self.photos, Qt.BottomDockWidgetArea)
        self.console_dock = self._dock("Console", self.console, Qt.BottomDockWidgetArea)
        self.tabifyDockWidget(self.photos_dock, self.console_dock)
        self.photos_dock.raise_()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: (self.resizeDocks([self.ws_dock, self.ref_dock], [max(320, self.width() // 4)] * 2, Qt.Horizontal),
                                      self.resizeDocks([self.photos_dock], [230], Qt.Vertical)))

        self.bridge = LogBridge(); self.bridge.line.connect(self.console.append_line)
        self.bridge.line.connect(self._status_from_log)
        self._task_title = None
        util.LOG_HOOKS.append(self.bridge.line.emit)

        self._build_actions()
        self.status_lbl = QLabel(); self.statusBar().addWidget(self.status_lbl, 1)
        self.busy = QProgressBar(); self.busy.setMaximumWidth(220); self.busy.setRange(0, 0); self.busy.hide()
        self.statusBar().addPermanentWidget(self.busy)

        self.workspace.layer_activated.connect(self.set_display)
        self.workspace.chunk_activated.connect(self.set_chunk)
        self.workspace.chunk_action.connect(self.chunk_action)
        self.photos.open_photo.connect(self.open_photo)
        self.photos.camera_selected.connect(self.model.highlight_camera)
        self.photos.cameras_enabled.connect(self._enable_cameras)
        self.reference.changed.connect(self._reference_changed)
        self.reference.request.connect(self._reference_request)
        self.reference.marker_selected.connect(self._marker_selected)
        self.model.picked.connect(self._picked)
        self.model.selection_changed.connect(lambda layer, n: self.status("%s selected: %s" % (layer, "{:,}".format(n))))
        self.model.region_changed.connect(lambda r: self.save())

        if project_dir:
            self.open_project(project_dir)
        else:
            last = self.settings.value("last_project")
            if last and os.path.exists(os.path.join(last, Project.FILE)):
                self.open_project(last)

    # ------------------------------------------------------------------ ui construction
    def _dock(self, title, w, area):
        d = QDockWidget(title, self); d.setObjectName(title); d.setWidget(w)
        self.addDockWidget(area, d)
        return d

    def _act(self, menu, text, fn, shortcut=None, checkable=False, tip=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        if tip:
            a.setStatusTip(tip)
        a.triggered.connect(fn)
        if menu is not None:
            menu.addAction(a)
        return a

    def _build_actions(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        self._act(m, "New Project…", self.new_project, "Ctrl+N")
        self._act(m, "Open Project…", self.open_project_dialog, "Ctrl+O")
        self._act(m, "Save", self.save, "Ctrl+S")
        m.addSeparator()
        ex = m.addMenu("Export")
        self._act(ex, "Export Model…", self.export_model)
        self._act(ex, "Export Point Cloud…", lambda: self.export_points("dense"))
        self._act(ex, "Export Tie Points…", lambda: self.export_points("sparse"))
        self._act(ex, "Export Cameras (COLMAP)…", self.export_cameras)
        self._act(ex, "Export Markers (CSV)…", self.export_markers)
        self._act(ex, "Fly-through Video…", self.flythrough)
        self._act(m, "Generate Report", self.report)
        m.addSeparator()
        self._act(m, "Exit", self.close, "Ctrl+Q")

        w = mb.addMenu("&Workflow")
        self.workflow_actions = [
            self._act(w, "Import Video…", self.import_video),
            self._act(w, "Add Photos…", self.add_photos),
        ]
        w.addSeparator()
        self.workflow_actions += [
            self._act(w, "Align Photos…", self.align),
            self._act(w, "Build Depth Maps…", self.depth),
            self._act(w, "Build Point Cloud", self.cloud),
            self._act(w, "Build Mesh…", self.mesh),
            self._act(w, "Build Texture…", self.texture),
        ]
        w.addSeparator()
        self.workflow_actions.append(self._act(w, "Batch Process (all steps)…", self.process_all))
        w.addSeparator()
        self.workflow_actions += [
            self._act(w, "Add Chunk", lambda: self.chunk_action("add", -1)),
            self._act(w, "Align Chunks…", lambda: self.chunk_action("align", -1)),
            self._act(w, "Merge Chunks…", lambda: self.chunk_action("merge", -1)),
        ]

        t = mb.addMenu("&Tools")
        self.workflow_actions += [
            self._act(t, "Estimate Image Quality", self.quality),
            self._act(t, "Detect Markers…", lambda: self._reference_request("detect")),
            self._act(t, "Optimize Cameras…", self.optimize),
        ]
        self._act(t, "Camera Calibration…", self.camera_calibration)
        t.addSeparator()
        mm = t.addMenu("Masks")
        self.workflow_actions += [
            self._act(mm, "Generate Masks (AI)…", self.masks_ai),
            self._act(mm, "Import Masks…", self.masks_import),
        ]
        self._act(mm, "Clear All Masks", self.masks_clear)
        self.workflow_actions.append(self._act(t, "Fly-through Video…", self.flythrough))
        mt = t.addMenu("Mesh")
        self.workflow_actions += [
            self._act(mt, "Refine Mesh…", self.refine_mesh),
            self._act(mt, "Close Holes…", self.close_holes),
        ]
        t.addSeparator()
        self._act(t, "Gradual Selection (tie points)…", self.gradual_selection)
        self._act(t, "Filter Point Cloud by Confidence…", self.filter_confidence)
        self._act(t, "Select Small Mesh Components…", self.select_components)
        t.addSeparator()
        self._act(t, "Reset Region", self.reset_region)
        t.addSeparator()
        self._act(t, "FFmpeg Location…", self.ffmpeg_settings,
                  tip="Choose which ffmpeg.exe / ffprobe.exe is used for video import and fly-through videos")

        v = mb.addMenu("&View")
        self.display_group = QActionGroup(self)
        for k, name in enumerate(DISPLAY_MODES):
            a = self._act(v, name, lambda _=False, n=name: self.set_display(n), "%d" % (k + 1), checkable=True)
            self.display_group.addAction(a)
            if name == self.model.display:
                a.setChecked(True)
        v.addSeparator()
        self.a_cams = self._act(v, "Show Cameras", self._toggle_layers, checkable=True); self.a_cams.setChecked(True)
        self.a_region = self._act(v, "Show Region", self._toggle_layers, checkable=True); self.a_region.setChecked(True)
        self.a_markers = self._act(v, "Show Markers", self._toggle_layers, checkable=True); self.a_markers.setChecked(True)
        v.addSeparator()
        self._act(v, "Reset View", self.model.reset_view, "0")
        self._act(v, "Top", lambda: self.model.view_from("top"), "7")
        self._act(v, "Front", lambda: self.model.view_from("front"), "8")
        self._act(v, "Side", lambda: self.model.view_from("side"), "9")

        tb = QToolBar("Tools"); tb.setObjectName("tools"); self.addToolBar(tb)
        self.mode_group = QActionGroup(self)
        self.tool_actions = {}
        for key, text, sc, tip in [("navigate", "Navigate", "Space", "Rotate: left drag · pan: middle drag · zoom: wheel"),
                                    ("rect", "Rect Select", "R", "Drag a rectangle. Shift = add, Ctrl = subtract"),
                                    ("lasso", "Free-form Select", "L", "Draw a free-form outline. Shift = add, Ctrl = subtract"),
                                    ("pick", "Add Marker", "M", "Click on the model to create a marker there"),
                                    ("measure", "Ruler", "U", "Click two points to measure the distance")]:
            a = self._act(None, text, lambda _=False, k=key: self.set_tool(k), sc, checkable=True, tip=tip)
            self.mode_group.addAction(a); tb.addAction(a); self.tool_actions[key] = a
        self.tool_actions["navigate"].setChecked(True)
        tb.addSeparator()
        self._act(tb, "Delete Selection", self.delete_selection, "Del", tip="Delete selected points / faces")
        self._act(tb, "Crop Selection", self.crop_selection, "Ctrl+Shift+C", tip="Keep only the selection")
        self._act(tb, "Clear Selection", self.model.clear_selection, "Esc")
        tb.addSeparator()
        self.a_region_edit = self._act(tb, "Edit Region", self.toggle_region_edit, "B", checkable=True,
                                       tip="Move / scale / rotate the reconstruction bounding box")
        tb.addSeparator()
        self.display_box = QComboBox(); self.display_box.addItems(DISPLAY_MODES)
        self.display_box.setCurrentText(self.model.display)
        self.display_box.currentTextChanged.connect(self.set_display)
        tb.addWidget(QLabel(" Display: ")); tb.addWidget(self.display_box)

    # ------------------------------------------------------------------ project
    def status(self, s):
        self.status_lbl.setText(s)

    def new_project(self):
        d = QFileDialog.getExistingDirectory(self, "Choose an empty folder for the new project")
        if d:
            Project.create(d)
            self.open_project(d)

    def open_project_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Open project folder (contains project.json)")
        if d:
            self.open_project(d)

    def open_project(self, d):
        try:
            self.project = Project.open(d)
        except Exception as e:
            QMessageBox.critical(self, "Open project", "Could not open %s:\n%s" % (d, e)); return
        self.chunk_idx = 0
        self.chunk = self.project.chunk(0)
        self.settings.setValue("last_project", self.project.dir)
        self.setWindowTitle("shapekit — %s" % self.project.dir)
        for k in range(self.tabs.count() - 1, 0, -1):
            self.tabs.removeTab(k)
        self.refresh_all(reset_camera=True)

    def save(self):
        if self.project:
            self.project.save()
            self.status("Saved")

    def refresh_all(self, reset_camera=False, reload_project=False):
        if reload_project and self.project:
            self.project = Project.open(self.project.dir)
            self.chunk_idx = min(self.chunk_idx, len(self.project.chunks) - 1)
            self.chunk = self.project.chunk(self.chunk_idx)
        if self.chunk is None:
            return
        if self.chunk.stats.get("mesh") is None:
            mode = "Point cloud" if self.chunk.stats.get("point_cloud") else "Tie points"
            self.model.display = mode; self.display_box.blockSignals(True)
            self.display_box.setCurrentText(mode); self.display_box.blockSignals(False)
        elif "texture" not in self.chunk.stats and self.model.display == "Mesh (textured)":
            self.model.display = "Mesh (shaded)"
        self.model.set_chunk(self.chunk, reset_camera=reset_camera)
        self.workspace.refresh(self.project, self.chunk)
        self.photos.refresh(self.chunk)
        self.reference.refresh(self.chunk)
        for k in range(1, self.tabs.count()):
            self.tabs.widget(k).set_chunk(self.chunk)

    def set_display(self, name):
        self.model.display = name
        self.display_box.blockSignals(True); self.display_box.setCurrentText(name); self.display_box.blockSignals(False)
        for a in self.display_group.actions():
            a.setChecked(a.text() == name)
        self.model.reload()

    def _toggle_layers(self):
        self.model.show_cameras = self.a_cams.isChecked()
        self.model.show_region = self.a_region.isChecked()
        self.model.show_markers = self.a_markers.isChecked()
        self.model.reload()

    def set_tool(self, key):
        self.model.set_mode(key)
        self.tool_actions[key].setChecked(True)
        if key == "measure":
            self.model.measure_pts = []
            self.status("Ruler: click two points on the model")
        elif key == "pick":
            self.status("Add marker: click on the model surface")

    # ------------------------------------------------------------------ background tasks
    def run(self, title, fn, *a, done=None, reload=False, **k):
        if self.task is not None:
            QMessageBox.information(self, "Busy", "Another task is running."); return
        if self.project:
            self.project.save()
        for act in self.workflow_actions:
            act.setEnabled(False)
        self.busy.show(); self.status(title + "…"); self._task_title = title
        self.console_dock.raise_()
        self.task = Task(fn, *a, **k)

        def finished(res, err):
            self.task = None
            self._task_title = None
            self.busy.hide()
            for act in self.workflow_actions:
                act.setEnabled(True)
            if err:
                self.console.append_line(err)
                print("TASK FAILED: %s\n%s" % (title, err), flush=True)      # also into gui.log
                QMessageBox.critical(self, title, err.strip().splitlines()[-1])
                self.status(title + " failed")
            else:
                self.status(title + " finished")
            if self.project and not reload:
                try:
                    self.project.save()
                except Exception:
                    pass
            self.model.invalidate()
            self.refresh_all(reload_project=reload)
            if done and not err:
                done(res)
        self.task.finished_with.connect(finished)
        self.task.start()

    def _status_from_log(self, line):
        """While a task runs, the status bar shows its latest progress line (the console may be hidden)."""
        if self._task_title is None:
            return
        msg = line.split("] ", 1)[-1].strip()
        if msg:
            if len(msg) > 140:
                msg = msg[:137] + "…"
            self.status("%s…  —  %s" % (self._task_title, msg))

    def _need(self, what):
        if self.chunk is None:
            QMessageBox.information(self, "No project", "Create or open a project first."); return False
        if what == "aligned" and not self.chunk.aligned_cameras():
            QMessageBox.information(self, "Not aligned", "Align photos first."); return False
        return True

    # ------------------------------------------------------------------ workflow
    def import_video(self):
        if not self._need(None) or not self._ensure_ffmpeg():
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Import video", "", "Video (*.mp4 *.mov *.m4v *.avi *.mkv *.mts)")
        if not files:
            return
        v = ParamDialog.ask("Import Video", [
            ("shift", "Keyframe every … of image motion", "float", 0.08, (0.02, 0.5, 3)),
            ("gap", "Max seconds between keyframes", "float", 1.5, (0.2, 10, 1))], self,
            note="Frames are picked automatically: a new keyframe whenever the view moved by the given fraction of "
                 "the image width; within each window the sharpest frame wins.")
        if v is None:
            return
        from ..video import import_video

        def job():
            for f in files:
                import_video(self.chunk, f, v["shift"], v["gap"])
                self.project.save()
        self.run("Import video", job)

    def add_photos(self):
        if not self._need(None):
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Add photos", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if files:
            from ..video import add_photos, estimate_image_quality
            self.run("Add photos", lambda: (add_photos(self.chunk, files), estimate_image_quality(self.chunk)))

    def quality(self):
        if self._need(None):
            from ..video import estimate_image_quality
            self.run("Estimate image quality", estimate_image_quality, self.chunk)

    def align(self):
        if not self._need(None):
            return
        v = ParamDialog.ask("Align Photos", [
            ("features", "Features", "choice", "auto", FEATURES),
            ("accuracy", "Accuracy", "choice", "high", ["highest", "high", "medium", "low", "lowest"]),
            ("keypoints", "Key point limit (0 = default)", "int", 0, (0, 100000)),
            ("seq", "Sequential preselection window", "int", 12, (0, 200)),
            ("generic", "Generic preselection (loop-closure pairs/photo)", "int", 10, (0, 200)),
            ("retrieval", "Retrieval descriptor", "choice", "auto", ["auto", "salad", "dinov2", "vlad"]),
            ("cross", "Cross-video pairs per photo", "int", 6, (0, 100)),
            ("reset", "Reset current alignment", "bool", True, None),
            ("reset_features", "Recompute features && matches", "bool", False, None),
            ("markers", "Use coded targets / markers as tie points", "bool", True, None),
            ("ba", "GPU bundle adjustment precision", "choice", "double", ["double", "mixed"])], self,
            note="auto = ALIKED + LightGlue on the GPU when PyTorch is installed, otherwise SIFT. Features, matches "
                 "and an interrupted alignment are reused automatically unless you tick the reset boxes. "
                 "Mixed precision = float32 Jacobians on the GPU (faster, results differ slightly). Masks apply to "
                 "tie points. Fisheye lenses: set the camera type in Tools → Camera Calibration first.")
        if v is None:
            return
        from ..cli import resolve_features
        from ..sfm import align_photos
        method = resolve_features(v["features"])
        kp = v["keypoints"] or (10000 if method == "sift" else 4096)
        self.run("Align photos", align_photos, self.chunk, v["accuracy"], kp, v["seq"], v["generic"],
                 reset=v["reset"], method=method, reset_features=v["reset_features"],
                 retrieval=v["retrieval"], cross_k=v["cross"], use_markers=v["markers"], ba_precision=v["ba"])

    def optimize(self):
        if not self._need("aligned"):
            return
        v = ParamDialog.ask("Optimize Cameras", [
            ("maxerr", "Remove projections with error above (px, 0 = keep)", "float", 0.0, (0, 50, 2)),
            ("gcp", "Use markers with reference X/Y/Z as control points (GCPs)", "bool", True, None),
            ("ba", "GPU bundle adjustment precision", "choice", "double", ["double", "mixed"])], self,
            note="Markers are always included as weighted observations. With 3+ reference markers the shape is "
                 "also pulled onto the surveyed coordinates (corrects bending of long video sequences).")
        if v is not None:
            from ..sfm import optimize_cameras
            self.run("Optimize cameras", optimize_cameras, self.chunk, max_error_px=v["maxerr"] or None, gcp=v["gcp"],
                     ba_precision=v["ba"])

    def depth(self):
        if not self._need("aligned"):
            return
        v = ParamDialog.ask("Build Depth Maps", [
            ("quality", "Quality", "choice", "high", ["ultra", "high", "medium", "low", "lowest"]),
            ("filter", "Depth filtering", "choice", "moderate", ["none", "mild", "moderate", "aggressive"]),
            ("method", "Method", "choice", "plane sweep", ["plane sweep", "PatchMatch (slanted surfaces)"]),
            ("reset", "Recompute existing", "bool", True, None)], self,
            note="PatchMatch estimates a surface normal per pixel: better on ground seen at grazing angles, walls and "
                 "steps (no stair-casing); plane sweep is the fronto-parallel method used so far. Masked pixels get no depth.")
        if v is not None:
            from ..mvs import build_depth_maps
            self.run("Build depth maps", build_depth_maps, self.chunk, v["quality"], v["filter"], force=v["reset"],
                     method="patchmatch" if v["method"].startswith("PatchMatch") else "planesweep")

    def cloud(self):
        if self._need("aligned"):
            from ..mvs import build_point_cloud
            self.run("Build point cloud", build_point_cloud, self.chunk,
                     done=lambda _: self.set_display("Point cloud"))

    def mesh(self):
        if not self._need("aligned"):
            return
        v = ParamDialog.ask("Build Mesh", [("faces", "Face count", "choice", "medium", ["high", "medium", "low"])], self,
                            note="Built from the filtered depth maps inside the region.")
        if v is not None:
            from ..mesh import build_mesh
            self.run("Build mesh", build_mesh, self.chunk, v["faces"], done=lambda _: self.set_display("Mesh (shaded)"))

    def texture(self):
        if not self._need("aligned"):
            return
        v = ParamDialog.ask("Build Texture", [("size", "Texture size", "choice", "8192", ["2048", "4096", "8192", "16384"]),
                                              ("blending", "Blending mode", "choice", "mosaic", ["mosaic", "multiband"])], self,
                            note="Mosaic: every surface patch from its single best photo. Multiband: same detail, plus the "
                                 "low frequencies blended across all photos, which levels out seams and exposure steps.")
        if v is not None:
            from ..texture import build_texture
            self.run("Build texture", build_texture, self.chunk, int(v["size"]), blending=v["blending"],
                     done=lambda _: self.set_display("Mesh (textured)"))

    def process_all(self):
        if not self._need(None):
            return
        v = ParamDialog.ask("Batch Process", [
            ("preset", "Preset", "choice", "balanced", ["fast", "balanced", "quality"]),
            ("features", "Features", "choice", "auto", FEATURES),
            ("markers", "Detect coded targets", "bool", True, None),
            ("depth_method", "Depth maps method", "choice", "planesweep", ["planesweep", "patchmatch"]),
            ("close_holes", "Close mesh holes up to N edges (0 = off)", "int", 0, (0, 100000)),
            ("refine", "Refine mesh (photo-consistency)", "bool", False, None),
            ("blending", "Texture blending", "choice", "mosaic", ["mosaic", "multiband"]),
            ("fly", "Fly-through video", "choice", "none", ["none", "orbit", "helix", "flyover", "reveal", "topdown", "capture"])], self,
            note="Runs alignment (if needed), marker detection, depth maps, point cloud, mesh, texture, exports and report "
                 "on the active chunk.")
        if v is None:
            return
        from ..cli import cmd_process
        ns = types.SimpleNamespace(project=self.project.dir, chunk=self.chunk_idx, video=None, photos=None,
                                   preset=v["preset"], shift=0.08, max_gap=1.5, workers=None, no_cloud=False,
                                   features=v["features"], no_markers=not v["markers"], depth_method=v["depth_method"],
                                   close_holes=v["close_holes"], refine=v["refine"], blending=v["blending"],
                                   flythrough=None if v["fly"] == "none" else v["fly"])

        def job():
            self.project.save()
            cmd_process(ns)
        self.run("Batch process", job, reload=True)

    # ------------------------------------------------------------------ exports
    def _save_path(self, title, filt, default):
        p, _ = QFileDialog.getSaveFileName(self, title, os.path.join(self.project.dir, "exports", default), filt)
        return p

    def export_model(self):
        if not self._need(None):
            return
        p = self._save_path("Export model", "glTF binary (*.glb);;Wavefront OBJ (*.obj);;PLY (*.ply)", "model.glb")
        if p:
            from ..export import export_model
            self.run("Export model", export_model, self.chunk, p)

    def export_points(self, src):
        if not self._need(None):
            return
        p = self._save_path("Export points", "PLY (*.ply)", "point_cloud.ply" if src == "dense" else "tie_points.ply")
        if p:
            from ..export import export_points
            self.run("Export points", export_points, self.chunk, p, src)

    def export_cameras(self):
        if not self._need("aligned"):
            return
        d = QFileDialog.getExistingDirectory(self, "Export COLMAP model to folder")
        if d:
            from ..export import export_cameras
            self.run("Export cameras", export_cameras, self.chunk, d)

    def export_markers(self):
        if not self._need(None):
            return
        p = self._save_path("Export markers", "CSV (*.csv)", "markers.csv")
        if p:
            from ..export import export_markers
            export_markers(self.chunk, p)

    def report(self):
        if not self._need(None):
            return
        from ..report import generate_report
        self.run("Generate report", generate_report, self.chunk,
                 done=lambda path: QDesktopServices.openUrl(QUrl.fromLocalFile(path)))

    # ------------------------------------------------------------------ photos
    def open_photo(self, i):
        for k in range(1, self.tabs.count()):
            if self.tabs.widget(k).cam == i:
                self.tabs.setCurrentIndex(k); return
        pv = PhotoView(self.chunk, i)
        pv.marker_moved.connect(lambda l, x, y, i=i: self._place(l, i, x, y))
        pv.marker_placed.connect(lambda l, x, y, i=i: self._place(l, i, x, y))
        pv.marker_removed.connect(lambda l, i=i: self._remove_projection(l, i))
        pv.mask_changed.connect(lambda i: (self.workspace.refresh(self.project, self.chunk),
                                           self.status("Mask saved for %s" % self.chunk.cameras[i].label)))
        self.tabs.addTab(pv, self.chunk.cameras[i].label)
        self.tabs.setCurrentWidget(pv)
        self.model.highlight_camera(i)

    def _close_tab(self, k):
        if k > 0:
            self.tabs.removeTab(k)

    def _enable_cameras(self, idx, on):
        for i in idx:
            self.chunk.cameras[i].enabled = on
        self.save(); self.photos.refresh(self.chunk)

    # ------------------------------------------------------------------ markers / reference
    def _place(self, label, i, x, y):
        ch = self.chunk
        if not label:
            label, ok = QInputDialog.getText(self, "New marker", "Marker label:",
                                             text="point %d" % (len(ch.markers or []) + 1))
            if not ok or not label:
                return
            if MK.find_marker(ch, label) is None:
                MK.add_marker(ch, label)
        m = MK.find_marker(ch, label)
        m["projections"][str(i)] = dict(xy=[x, y], pinned=True)
        MK.triangulate_marker(ch, m)
        self._markers_changed()

    def _remove_projection(self, label, i):
        m = MK.find_marker(self.chunk, label)
        if m and str(i) in m["projections"]:
            del m["projections"][str(i)]
            MK.triangulate_marker(self.chunk, m)
            self._markers_changed()

    def _markers_changed(self):
        if self.chunk.transform is not None:
            try:
                MK.update_transform(self.chunk)
            except Exception:
                pass
        self.save()
        self.reference.refresh(self.chunk)
        self.workspace.refresh(self.project, self.chunk)
        self.model.reload()
        for k in range(1, self.tabs.count()):
            self.tabs.widget(k).redraw()

    def _reference_changed(self):
        self._markers_changed()

    def _marker_selected(self, label):
        i = self.photos.filter.findText("With marker: " + label)
        if i >= 0:
            self.photos.filter.setCurrentIndex(i)

    def _reference_request(self, key):
        ch = self.chunk
        if ch is None:
            return
        if key == "detect":
            if not self._need("aligned"):
                return
            v = ParamDialog.ask("Detect Markers", [
                ("bits", "Target type", "choice", "12", ["12", "14", "16", "20"]),
                ("min_radius", "Minimum centre-dot radius (px)", "float", 2.0, (1, 50, 1)),
                ("tolerance", "Decoding tolerance (higher = stricter)", "float", 0.3, (0.05, 0.9, 2)),
                ("guided", "Guided placement in all photos", "bool", True, None)], self,
                note="Circular coded targets (centre dot + coded ring). IDs seen in fewer than 2 photos or "
                     "inconsistent across photos are rejected.")
            if v is None:
                return

            def job():
                ch.markers = [m for m in ch.markers if not m.get("auto")]
                MK.detect_markers(ch, int(v["bits"]), v["min_radius"], v["tolerance"])
                if v["guided"]:
                    MK.refine_marker_projections(ch)
            self.run("Detect markers", job)
        elif key == "optimize_gcp":
            if self._need("aligned"):
                from ..sfm import optimize_cameras
                self.run("Optimize cameras with GCPs", optimize_cameras, ch, gcp=True)
        elif key == "guided":
            if self._need("aligned"):
                self.run("Guided marker placement", MK.refine_marker_projections, self.chunk)
        elif key == "update":
            try:
                rep = MK.update_transform(ch)
            except Exception as e:
                QMessageBox.information(self, "Update transform", str(e)); return
            self._markers_changed()
            self.model.invalidate(); self.model.reload()
            self.status("Coordinate system updated: %s, scale %.6g" % (rep["mode"], rep["scale"]))
        elif key == "import":
            p, _ = QFileDialog.getOpenFileName(self, "Import reference CSV (label,x,y,z[,accuracy])", "", "CSV (*.csv *.txt)")
            if p:
                import csv
                n = 0
                for row in csv.reader(open(p)):
                    if len(row) >= 4 and MK.find_marker(ch, row[0].strip()) is not None:
                        try:
                            MK.set_reference(ch, row[0].strip(), [float(x) for x in row[1:4]],
                                             float(row[4]) if len(row) > 4 and row[4] else 0.005)
                            n += 1
                        except ValueError:
                            pass
                self.status("%d reference coordinates imported — press Update" % n)
                self._markers_changed()
        elif key == "add_scalebar":
            sel = self.reference.selected_markers()
            if len(sel) != 2:
                QMessageBox.information(self, "Add scale bar", "Select exactly two markers in the Markers table "
                                        "(Ctrl+click), then press Add scale bar."); return
            d, ok = QInputDialog.getDouble(self, "Scale bar", "Distance %s – %s (metres):" % tuple(sel), 1.0, 0, 1e6, 4)
            if ok:
                MK.add_scalebar(ch, sel[0], sel[1], d)
                try:
                    MK.update_transform(ch)
                except Exception as e:
                    self.status(str(e))
                self._markers_changed(); self.model.invalidate(); self.model.reload()
        elif key == "remove":
            if self.reference.tabs.currentIndex() == 0:
                labels = set(self.reference.selected_markers())
                ch.markers = [m for m in ch.markers if m["label"] not in labels]
                ch.scalebars = [b for b in ch.scalebars if b["a"] not in labels and b["b"] not in labels]
            else:
                rows = set(self.reference.selected_scalebars())
                ch.scalebars = [b for k, b in enumerate(ch.scalebars) if k not in rows]
            self._markers_changed()

    def _unique_label(self, label):
        base, k = label, 2
        while MK.find_marker(self.chunk, label) is not None:
            label = "%s (%d)" % (base, k); k += 1
        return label

    def _picked(self, X):
        if self.model.mode == "pick":
            label, ok = QInputDialog.getText(self, "New marker", "Marker label:",
                                             text="point %d" % (len(self.chunk.markers or []) + 1))
            if ok and label:
                label = self._unique_label(label)
                MK.add_marker(self.chunk, label, position=X)
                self._markers_changed()
                self.status("Marker '%s' created — refine it by dragging its blue estimate in a few photos" % label)
        elif self.model.mode == "measure":
            pts = self.model.measure_pts
            if len(pts) >= 2:
                pts.clear()
            pts.append(X)
            self.model.draw_measure(pts)
            if len(pts) == 2:
                d = MK.measure_distance(self.chunk, pts[0], pts[1])
                unit = "m" if self.chunk.transform is not None else "units (no scale set)"
                self.status("Distance: %.4f %s" % (d, unit))

    # ------------------------------------------------------------------ editing
    def _selection_or_warn(self):
        layer, mask = self.model.selection()
        if mask is None or not mask.any():
            self.status("Nothing selected — use Rect Select (R) or Free-form Select (L)")
            return None, None
        return layer, mask

    def delete_selection(self, keep=False):
        layer, mask = self._selection_or_warn()
        if layer is None:
            return
        m = ~mask if keep else mask
        n = int(m.sum())
        if QMessageBox.question(self, "Delete", "Delete %s %s?" % ("{:,}".format(n),
                                {"tie": "tie points", "dense": "points", "mesh": "faces"}[layer])) != QMessageBox.Yes:
            return
        if layer == "tie":
            edit.delete_tie_points(self.chunk, m)
        elif layer == "dense":
            if (self.model.data.get("dense") or {}).get("stride", 1) > 1:
                QMessageBox.information(self, "Delete points", "The view shows a subsample of this very large point "
                                        "cloud, so points can't be deleted one by one. Crop with the region box and "
                                        "rebuild the point cloud instead.")
                return
            edit.delete_dense_points(self.chunk, m)
        else:
            edit.delete_faces(self.chunk, m)
        self.save()
        self.model.invalidate(layer); self.model.reload()
        self.workspace.refresh(self.project, self.chunk)

    def crop_selection(self):
        self.delete_selection(keep=True)

    def gradual_selection(self):
        if not self._need("aligned"):
            return
        v = ParamDialog.ask("Gradual Selection", [
            ("crit", "Criterion", "choice", "Reprojection error", ["Reprojection error", "Image count"]),
            ("thr", "Level", "float", 1.0, (0, 1000, 2))], self,
            note="Selects tie points with reprojection error ABOVE the level, or seen in FEWER images than the level. "
                 "Then press Delete and run Tools → Optimize Cameras.")
        if v is None:
            return
        if v["crit"] == "Image count":
            vals = edit.tie_point_errors(self.chunk, "image_count"); mask = vals < v["thr"]
        else:
            vals = edit.tie_point_errors(self.chunk); mask = vals > v["thr"]
        self.set_display("Tie points")
        self.model.set_selection("tie", mask)

    def filter_confidence(self):
        if not self.chunk or not self.chunk.stats.get("point_cloud"):
            return
        v = ParamDialog.ask("Filter by confidence", [("c", "Select points with confidence below", "int", 3, (1, 255))], self)
        if v is None:
            return
        from ..io import read_ply, ply_vertex_count
        if ply_vertex_count(self.chunk.path("point_cloud.ply")) > 30_000_000:
            QMessageBox.information(self, "Filter by confidence", "This point cloud is too large to edit point by point. "
                                    "Use the region box (Edit Region) and rebuild, or build the mesh from depth maps.")
            return
        d = read_ply(self.chunk.path("point_cloud.ply"))
        if "confidence" in d:
            self.set_display("Point cloud")
            self.model.set_selection("dense", d["confidence"] < v["c"])

    def select_components(self):
        if not self.chunk or not self.chunk.stats.get("mesh"):
            return
        v = ParamDialog.ask("Select components", [("f", "Smaller than (fraction of largest)", "float", 0.05, (0, 1, 3))], self)
        if v is not None:
            self.set_display("Mesh (shaded)")
            self.model.set_selection("mesh", edit.small_component_faces(self.chunk, v["f"]))

    def toggle_region_edit(self):
        on = self.a_region_edit.isChecked()
        self.model.edit_region(on)
        self.status("Region editing: drag faces / handles to move, scale and rotate; toggle Edit Region off to apply"
                    if on else "Region applied — rebuild the mesh to use it")

    def reset_region(self):
        if self._need("aligned"):
            from ..sfm import region_from_sparse
            region_from_sparse(self.chunk); self.save(); self.model.reload()

    # ------------------------------------------------------------------ chunks
    def set_chunk(self, idx):
        if self.project is None or not (0 <= idx < len(self.project.chunks)):
            return
        if self.task is not None:
            QMessageBox.information(self, "Busy", "Wait for the running task to finish."); return
        self.save()
        self.chunk_idx = idx
        self.chunk = self.project.chunk(idx)
        for k in range(self.tabs.count() - 1, 0, -1):
            self.tabs.removeTab(k)
        self.model.invalidate()
        self.refresh_all(reset_camera=True)
        self.status("Active chunk: %s" % self.chunk.label)

    def chunk_action(self, action, idx):
        from .. import chunks as CK
        if self.project is None:
            return
        P = self.project
        labels = ["%d: %s" % (k, c.label) for k, c in enumerate(P.chunks)]
        if action == "add":
            ch = CK.add_chunk(P); self.save(); self.set_chunk(len(P.chunks) - 1)
            self.status("Added '%s' — import photos or videos into it" % ch.label)
        elif action == "rename":
            s, ok = QInputDialog.getText(self, "Rename chunk", "Label:", text=P.chunks[idx].label)
            if ok and s:
                P.chunks[idx].label = s; self.save(); self.workspace.refresh(P, self.chunk)
        elif action == "duplicate":
            v = ParamDialog.ask("Duplicate Chunk", [
                ("label", "Label", "text", P.chunks[idx].label + " copy", None),
                ("alignment", "Copy alignment", "bool", True, None),
                ("depth", "Copy depth maps", "bool", True, None),
                ("cloud", "Copy point cloud", "bool", True, None),
                ("mesh", "Copy model and texture", "bool", True, None)], self,
                note="Photos are referenced, not copied.")
            if v is None:
                return
            items = ["features"] + [k for k in ("alignment", "depth", "cloud") if v[k]] + (["mesh", "texture"] if v["mesh"] else [])
            self.run("Duplicate chunk", CK.duplicate_chunk, P, idx, v["label"], items, reload=False,
                     done=lambda _: self.workspace.refresh(P, self.chunk))
        elif action == "remove":
            if len(P.chunks) == 1:
                QMessageBox.information(self, "Remove chunk", "A project needs at least one chunk."); return
            if QMessageBox.question(self, "Remove chunk", "Remove '%s' from the project? (its folder is kept on disk, "
                                    "renamed *_removed)" % P.chunks[idx].label) != QMessageBox.Yes:
                return
            CK.remove_chunk(P, idx)
            new = 0 if idx == self.chunk_idx else (self.chunk_idx - 1 if idx < self.chunk_idx else self.chunk_idx)
            self.chunk_idx = new; self.chunk = P.chunk(new)
            self.save(); self.set_chunk(new)
        elif action == "align":
            if len(P.chunks) < 2:
                QMessageBox.information(self, "Align chunks", "Add a second chunk first."); return
            v = ParamDialog.ask("Align Chunks", [
                ("ref", "Reference chunk", "choice", labels[self.chunk_idx], labels),
                ("by", "Method", "choice", "markers", ["markers", "cameras"])], self,
                note="All other chunks are moved into the reference chunk's coordinate system. markers = markers with "
                     "the same label in both chunks (>= 3, e.g. coded targets); cameras = the same photos aligned in both.")
            if v is None:
                return
            ref = labels.index(v["ref"])
            try:
                res = CK.align_chunks(P, ref, [k for k in range(len(P.chunks)) if k != ref], v["by"])
            except Exception as e:
                QMessageBox.information(self, "Align chunks", str(e)); return
            self.save(); self.refresh_all()
            self.status("Aligned: " + "; ".join("%s by %d (RMSE %.4g)" % (P.chunks[k].label, r["n"], r["rmse"])
                                                for k, r in res.items()))
        elif action == "merge":
            if len(P.chunks) < 2:
                QMessageBox.information(self, "Merge chunks", "Add a second chunk first."); return
            fields = [("c%d" % k, l, "bool", True, None) for k, l in enumerate(labels)]
            fields += [("label", "New chunk label", "text", "Merged Chunk", None),
                       ("depth", "Merge depth maps", "bool", True, None),
                       ("cloud", "Merge point clouds", "bool", True, None),
                       ("mesh", "Merge models", "bool", False, None)]
            v = ParamDialog.ask("Merge Chunks", fields, self,
                                note="Chunks should be aligned first (Align Chunks or reference markers). The result is a "
                                     "new chunk in the first selected chunk's frame; build a mesh from the merged depth maps.")
            if v is None:
                return
            idxs = [k for k in range(len(P.chunks)) if v["c%d" % k]]
            if len(idxs) < 2:
                return
            self.run("Merge chunks", CK.merge_chunks, P, idxs, v["label"], True, v["depth"], v["cloud"], v["mesh"],
                     done=lambda _: self.set_chunk(len(P.chunks) - 1))

    def flythrough(self):
        if not self._need(None):
            return
        st = self.chunk.stats
        if not (st.get("mesh") or st.get("point_cloud") or st.get("align")):
            QMessageBox.information(self, "Fly-through", "Build a model (or at least align the photos) first."); return
        if not self._ensure_ffmpeg():
            return
        v = ParamDialog.ask("Fly-through Video", [
            ("pattern", "Path", "choice", "orbit", ["orbit", "helix", "flyover", "reveal", "topdown", "capture"]),
            ("seconds", "Duration (s)", "float", 20.0, (2, 600, 1)),
            ("size", "Resolution", "choice", "1080p", ["480p", "720p", "1080p", "1440p", "4k"]),
            ("fps", "Frames per second", "choice", "30", ["24", "25", "30", "60"]),
            ("elevation", "Camera elevation (deg)", "float", 30.0, (0, 89, 1)),
            ("distance", "Distance (1 = site fills the frame)", "float", 1.0, (0.2, 5, 2)),
            ("arc", "Orbit sweep (deg; 180 suits sites filmed from one side)", "float", 360.0, (10, 360, 0)),
            ("display", "Show", "choice", "auto", ["auto", "textured", "colours", "shaded", "points"])], self,
            note="orbit = circle around the site; helix = descending spiral; flyover = straight pass over it; reveal = "
                 "start close and pull back; topdown = rotating bird's-eye view; capture = follows the path of your "
                 "photos/video. Paths start on the side the photos were taken from. Saved as MP4 (+ JPG) in exports.")
        if v is None:
            return
        out = os.path.join(self.project.dir, "exports", "flythrough_%s.mp4" % v["pattern"])
        from ..flythrough import run_external
        self.run("Fly-through video", run_external, self.project.dir, self.chunk_idx, out,
                 pattern=v["pattern"], seconds=v["seconds"], size=v["size"], fps=v["fps"], elevation=v["elevation"],
                 distance=v["distance"], arc=v["arc"], display=v["display"],
                 done=lambda path: QDesktopServices.openUrl(QUrl.fromLocalFile(path)))

    # ------------------------------------------------------------------ FFmpeg
    def _ensure_ffmpeg(self):
        """True if ffmpeg + ffprobe are found; otherwise offer to locate / download them."""
        from .. import ffmpeg as FF
        if FF.find("ffmpeg") and FF.find("ffprobe"):
            return True
        self.ffmpeg_settings(missing=True)
        return bool(FF.find("ffmpeg") and FF.find("ffprobe"))

    def ffmpeg_settings(self, checked=False, missing=False):
        from .. import ffmpeg as FF
        while True:
            FF.reset_cache()
            box = QMessageBox(self)
            box.setWindowTitle("FFmpeg Location")
            box.setIcon(QMessageBox.Warning if missing else QMessageBox.Information)
            head = ("FFmpeg is needed to import videos and render fly-throughs, and it was not found.\n\n"
                    if missing else "")
            box.setText(head + FF.describe())
            box.setInformativeText("Browse: pick ffmpeg.exe (ffprobe.exe must be in the same folder).\n"
                                   "Download: fetch a copy into the app's ffmpeg folder (about 100 MB).\n"
                                   "Automatic: forget the chosen location and search bundled copy / PATH / "
                                   "winget, Chocolatey, Scoop installs.")
            b_browse = box.addButton("Browse…", QMessageBox.ActionRole)
            b_dl = box.addButton("Download", QMessageBox.ActionRole) if sys.platform == "win32" else None
            b_auto = box.addButton("Automatic", QMessageBox.ResetRole)
            box.addButton(QMessageBox.Close)
            box.exec()
            b = box.clickedButton()
            if b is b_browse:
                start = os.path.dirname(FF.find("ffmpeg") or "") or os.environ.get("ProgramFiles", "")
                f, _ = QFileDialog.getOpenFileName(self, "Select ffmpeg.exe", start,
                                                   "ffmpeg (ffmpeg.exe ffmpeg);;Programs (*.exe);;All files (*)")
                if f:
                    try:
                        ff, fp = FF.set_location(f)
                        self.status("FFmpeg: %s" % ff)
                        missing = not (ff and fp)
                    except Exception as e:
                        QMessageBox.warning(self, "FFmpeg Location", str(e))
                continue
            if b is b_auto:
                FF.set_location(None)
                missing = not (FF.find("ffmpeg") and FF.find("ffprobe"))
                continue
            if b_dl is not None and b is b_dl:
                self._download_ffmpeg()
            return

    def _download_ffmpeg(self):
        from .. import ffmpeg as FF
        dest = FF.bundled_dir()
        try:
            os.makedirs(dest, exist_ok=True)
            probe = os.path.join(dest, ".write_test"); open(probe, "w").close(); os.remove(probe)
        except OSError:                                   # e.g. app folder under Program Files
            dest = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "shapekit", "ffmpeg")

        def job():
            from ..tools.get_ffmpeg import main as get
            if get(dest) != 0:
                raise RuntimeError("FFmpeg download failed - see the Console, or use Browse to pick an installed ffmpeg.exe")
            if os.path.normcase(dest) != os.path.normcase(FF.bundled_dir()):
                FF.set_location(dest)
            FF.reset_cache()
            return FF.find("ffmpeg")
        self.run("Download FFmpeg", job, done=lambda p: p and self.status("FFmpeg: %s" % p))

    # ------------------------------------------------------------------ calibration / masks / mesh tools
    def camera_calibration(self):
        if not self._need(None):
            return
        ch = self.chunk
        models = ["frame", "fisheye_equidistant", "fisheye_equisolid"]
        fields = []
        for s, sen in enumerate(ch.sensors):
            fields += [("model%d" % s, "%s — camera type" % sen.label, "choice", sen.model, models),
                       ("f%d" % s, "%s — focal length (px)" % sen.label, "float", float(sen.intr[0]), (10, 1e6, 2)),
                       ("fixed%d" % s, "%s — fix calibration" % sen.label, "bool", sen.fixed, None)]
        v = ParamDialog.ask("Camera Calibration", fields, self,
                            note="Fisheye: GoPro / action cams / 360 lenses (equidistant is the common model). Changing "
                                 "the type resets f to a starting guess; alignment refines it. Re-align afterwards.")
        if v is None:
            return
        for s, sen in enumerate(ch.sensors):
            changed = v["model%d" % s] != sen.model
            sen.set_model(v["model%d" % s])
            if not changed:
                sen.intr[0] = v["f%d" % s]
            sen.fixed = v["fixed%d" % s]
        self.save(); self.workspace.refresh(self.project, ch)

    def masks_ai(self):
        if not self._need(None):
            return
        v = ParamDialog.ask("Generate Masks (AI)", [
            ("people", "People", "bool", True, None), ("vehicles", "Vehicles", "bool", True, None),
            ("animals", "Animals", "bool", False, None),
            ("dilate", "Grow masks by (px)", "int", 12, (0, 200)),
            ("keep", "Keep existing (manual) masks", "bool", True, None)], self,
            note="Semantic segmentation (DeepLabV3) masks moving things so they don't create tie points, depth or "
                 "texture. The network (~170 MB) is downloaded on first use. Edit masks in the photo view.")
        if v is None:
            return
        groups = [g for g in ("people", "vehicles", "animals") if v[g]]
        from ..masks import generate_ai_masks
        self.run("Generate masks", generate_ai_masks, self.chunk, groups, v["dilate"], None, v["keep"])

    def masks_import(self):
        if not self._need(None):
            return
        d = QFileDialog.getExistingDirectory(self, "Folder with mask images")
        if not d:
            return
        v = ParamDialog.ask("Import Masks", [
            ("pattern", "File name pattern", "text", "{label}_mask.png", None),
            ("invert", "White = masked (invert)", "bool", False, None)], self,
            note="{label} = photo name without extension, {filename} = with extension. Black = masked by default.")
        if v is not None:
            from ..masks import import_masks
            self.run("Import masks", import_masks, self.chunk, d, v["pattern"], v["invert"])

    def masks_clear(self):
        if not self._need(None):
            return
        from .. import masks as MS
        n = len(MS.masked_cameras(self.chunk))
        if n and QMessageBox.question(self, "Clear masks", "Delete the masks of %d photos?" % n) == QMessageBox.Yes:
            for i in MS.masked_cameras(self.chunk):
                MS.clear_mask(self.chunk, i)
            self.refresh_all()

    def refine_mesh(self):
        if not self._need("aligned") or not self.chunk.stats.get("mesh"):
            return
        v = ParamDialog.ask("Refine Mesh", [("it", "Iterations", "int", 3, (1, 10))], self,
                            note="Moves every vertex along its normal to where the photos agree best (NCC), smoothed over "
                                 "the mesh. The original is kept as mesh_unrefined.ply. Rebuild the texture afterwards.")
        if v is not None:
            from ..mesh import refine_mesh
            self.run("Refine mesh", refine_mesh, self.chunk, v["it"], done=lambda _: self.set_display("Mesh (shaded)"))

    def close_holes(self):
        if not self._need(None) or not self.chunk.stats.get("mesh"):
            return
        v = ParamDialog.ask("Close Holes", [("n", "Fill holes with up to N border edges", "int", 200, (3, 1000000))], self,
                            note="Holes are filled with a smooth membrane. The outer border of the model is never closed.")
        if v is not None:
            from ..mesh import close_holes_chunk
            self.run("Close holes", close_holes_chunk, self.chunk, v["n"], done=lambda _: self.set_display("Mesh (shaded)"))

    # ------------------------------------------------------------------
    def closeEvent(self, ev):
        if self.task is not None:
            if QMessageBox.question(self, "Quit", "A task is still running. Quit anyway?") != QMessageBox.Yes:
                ev.ignore(); return
        try:
            util.LOG_HOOKS.remove(self.bridge.line.emit)
        except ValueError:
            pass
        self.save()
        self.model.close()
        super().closeEvent(ev)


def _crash_reporting():
    """Hard crashes (GPU driver, out of memory in native code) kill the process without a Python traceback.
    faulthandler writes the stack of every thread to crash.log when that happens; uncaught Python errors on
    the GUI thread go there too."""
    import faulthandler
    import traceback
    import time as _t
    d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "ShapeKit")
    try:
        os.makedirs(d, exist_ok=True)
        f = open(os.path.join(d, "crash.log"), "a", buffering=1, encoding="utf-8")
    except OSError:
        return None
    f.write("\n=== shapekit started %s ===\n" % _t.strftime("%Y-%m-%d %H:%M:%S"))
    faulthandler.enable(file=f, all_threads=True)

    def hook(et, ev, tb):
        f.write("Uncaught exception:\n" + "".join(traceback.format_exception(et, ev, tb)))
        sys.__excepthook__(et, ev, tb)
    sys.excepthook = hook
    return f.name


def run(project_dir=None):
    crash = _crash_reporting()
    if crash:
        print("crash reports (if the app ever closes unexpectedly): %s" % crash, flush=True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("shapekit")
    app.setStyle("Fusion")
    w = MainWindow(project_dir)
    w.show()
    return app.exec()
