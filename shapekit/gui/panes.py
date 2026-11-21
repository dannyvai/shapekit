"""Dock panes: Workspace, Photos, Reference (markers + scale bars), Console."""
import os
import numpy as np
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QIcon, QPixmap, QImageReader, QPainter, QColor, QBrush, QPen
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QListWidget,
                               QListWidgetItem, QListView, QTableWidget, QTableWidgetItem, QPushButton, QPlainTextEdit,
                               QTabWidget, QAbstractItemView, QHeaderView, QLabel, QMenu, QComboBox)


# ----------------------------------------------------------------------------- workspace
class WorkspacePane(QTreeWidget):
    layer_activated = Signal(str)
    chunk_activated = Signal(int)
    chunk_action = Signal(str, int)          # action, chunk index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Workspace", ""])
        self.setHeaderHidden(True)
        self.setMinimumWidth(120)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.itemDoubleClicked.connect(self._double)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def _double(self, it, _c):
        if it.data(0, Qt.UserRole + 1) is not None:
            self.chunk_activated.emit(int(it.data(0, Qt.UserRole + 1)))
        elif it.data(0, Qt.UserRole):
            self.layer_activated.emit(it.data(0, Qt.UserRole))

    def _menu(self, pos):
        it = self.itemAt(pos)
        idx = None
        while it is not None and idx is None:
            idx = it.data(0, Qt.UserRole + 1)
            it = it.parent()
        menu = QMenu(self)
        if idx is not None:
            idx = int(idx)
            menu.addAction("Set Active").triggered.connect(lambda: self.chunk_activated.emit(idx))
            menu.addAction("Rename…").triggered.connect(lambda: self.chunk_action.emit("rename", idx))
            menu.addAction("Duplicate…").triggered.connect(lambda: self.chunk_action.emit("duplicate", idx))
            menu.addAction("Remove").triggered.connect(lambda: self.chunk_action.emit("remove", idx))
            menu.addSeparator()
        menu.addAction("Add Chunk").triggered.connect(lambda: self.chunk_action.emit("add", -1))
        menu.addAction("Align Chunks…").triggered.connect(lambda: self.chunk_action.emit("align", -1))
        menu.addAction("Merge Chunks…").triggered.connect(lambda: self.chunk_action.emit("merge", -1))
        menu.exec(self.viewport().mapToGlobal(pos))

    def refresh(self, project, chunk):
        self.clear()
        if project is None:
            return
        root = QTreeWidgetItem([os.path.basename(project.dir), "%d chunk%s" % (len(project.chunks), "s" if len(project.chunks) != 1 else "")])
        self.addTopLevelItem(root)
        for k, c in enumerate(project.chunks):
            active = c is chunk
            ch = QTreeWidgetItem(root, [c.label + ("  (active)" if active and len(project.chunks) > 1 else ""),
                                        "%d/%d aligned" % (len(c.aligned_cameras()), len(c.cameras))])
            ch.setData(0, Qt.UserRole + 1, k)
            if active:
                f = ch.font(0); f.setBold(True); ch.setFont(0, f)
            else:
                continue
            st = c.stats

            def add(name, val, layer=None):
                it = QTreeWidgetItem(ch, [name, str(val)])
                if layer:
                    it.setData(0, Qt.UserRole, layer)
                return it
            add("Cameras", "%d/%d aligned" % (len(c.aligned_cameras()), len(c.cameras)))
            models = sorted({s.model for s in c.sensors})
            if models and models != ["frame"]:
                add("Camera model", ", ".join(models))
            try:
                from ..masks import masked_cameras
                nm = len(masked_cameras(c))
                if nm:
                    add("Masks", "%d photos" % nm)
            except Exception:
                pass
            if "align" in st:
                add("Tie points", "{:,}  ({:.2f} px)".format(st["align"].get("tie_points", 0),
                                                           st["align"].get("rms_reprojection_px", 0)), "Tie points")
            if "depth_maps" in st:
                add("Depth maps", "%s, %s%s" % (st["depth_maps"].get("quality"), st["depth_maps"].get("filter"),
                                                ", PatchMatch" if st["depth_maps"].get("method") == "patchmatch" else ""))
            if "point_cloud" in st:
                add("Point cloud", "{:,}".format(st["point_cloud"].get("points", 0)), "Point cloud")
            if "mesh" in st:
                add("3D model", "{:,} faces{}".format(st["mesh"].get("faces", 0), ", refined" if st["mesh"].get("refined") else ""),
                    "Mesh (shaded)")
            if "texture" in st:
                add("Texture", "%d px%s" % (st["texture"].get("size", 0),
                                            ", multiband" if st["texture"].get("blending") == "multiband" else ""), "Mesh (textured)")
            add("Markers", len(c.markers or []))
            add("Scale bars", len(c.scalebars or []))
            ref = st.get("reference")
            add("Coordinate system", ("scaled ×%.5g (%s)" % (ref["scale"], ref["mode"])) if c.transform and ref
                else "arbitrary (no scale)")
        self.expandAll()


# ----------------------------------------------------------------------------- photos
class PhotosPane(QWidget):
    open_photo = Signal(int)
    camera_selected = Signal(int)
    cameras_enabled = Signal(list, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(2, 2, 2, 2)
        top = QHBoxLayout()
        self.filter = QComboBox(); self.filter.addItem("All photos")
        self.filter.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon); self.filter.setMinimumContentsLength(12)
        self.filter.currentIndexChanged.connect(lambda _: self.apply_filter())
        top.addWidget(QLabel("Show:")); top.addWidget(self.filter, 1)
        lay.addLayout(top)
        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode); self.list.setIconSize(QSize(144, 81))
        self.list.setResizeMode(QListView.Adjust); self.list.setMovement(QListView.Static)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection); self.list.setUniformItemSizes(True)
        self.list.setSpacing(4)
        self.list.itemDoubleClicked.connect(lambda it: self.open_photo.emit(it.data(Qt.UserRole)))
        self.list.currentItemChanged.connect(lambda it, _p: it and self.camera_selected.emit(it.data(Qt.UserRole)))
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.list)
        self.chunk = None
        self._queue = []
        self._timer = QTimer(self); self._timer.timeout.connect(self._load_some)

    def refresh(self, chunk):
        self.chunk = chunk
        self.list.clear()
        cur = self.filter.currentText()
        self.filter.blockSignals(True); self.filter.clear(); self.filter.addItems(["All photos", "Not aligned", "Disabled"])
        for m in chunk.markers or []:
            self.filter.addItem("With marker: " + m["label"])
        i = self.filter.findText(cur); self.filter.setCurrentIndex(max(i, 0)); self.filter.blockSignals(False)
        ph = QPixmap(144, 81); ph.fill(QColor("#3a3f48"))
        for i, c in enumerate(chunk.cameras):
            it = QListWidgetItem(QIcon(ph), self._text(c))
            it.setData(Qt.UserRole, i)
            it.setToolTip("%s\nsource: %s frame %s\nquality: %s\n%s%s" % (
                c.label, c.source or "-", c.frame if c.frame is not None else "-", c.quality,
                "aligned" if c.aligned else "NOT aligned", "" if c.enabled else "\nDISABLED"))
            self.list.addItem(it)
        self._queue = list(range(len(chunk.cameras)))
        self._timer.start(0)
        self.apply_filter()

    def _text(self, c):
        flag = "✓" if c.aligned else "✗"
        return "%s %s\nq %.2f" % (flag, c.label[-10:], c.quality or 0)

    def _load_some(self):
        for _ in range(6):
            if not self._queue:
                self._timer.stop(); return
            i = self._queue.pop(0)
            c = self.chunk.cameras[i]
            r = QImageReader(self.chunk.abspath(c)); r.setScaledSize(QSize(144, 81))
            img = r.read()
            if img.isNull():
                continue
            pm = QPixmap.fromImage(img)
            p = QPainter(pm)
            col = QColor("#36e26a") if c.aligned else QColor("#ff4d5e")
            if not c.enabled:
                p.fillRect(pm.rect(), QColor(0, 0, 0, 150)); col = QColor("#999")
            p.setBrush(QBrush(col)); p.setPen(QPen(Qt.black, 1)); p.drawEllipse(4, 4, 10, 10); p.end()
            self.list.item(i).setIcon(QIcon(pm))

    def apply_filter(self):
        if self.chunk is None:
            return
        f = self.filter.currentText()
        lab = f[len("With marker: "):] if f.startswith("With marker: ") else None
        m = next((x for x in self.chunk.markers or [] if x["label"] == lab), None)
        for k in range(self.list.count()):
            it = self.list.item(k); c = self.chunk.cameras[it.data(Qt.UserRole)]
            show = True
            if f == "Not aligned":
                show = not c.aligned
            elif f == "Disabled":
                show = not c.enabled
            elif m is not None:
                show = str(it.data(Qt.UserRole)) in m["projections"]
            it.setHidden(not show)

    def selected(self):
        return [it.data(Qt.UserRole) for it in self.list.selectedItems()]

    def _menu(self, pos):
        sel = self.selected()
        if not sel:
            return
        menu = QMenu(self)
        menu.addAction("Open").triggered.connect(lambda: self.open_photo.emit(sel[0]))
        menu.addAction("Enable cameras").triggered.connect(lambda: self.cameras_enabled.emit(sel, True))
        menu.addAction("Disable cameras").triggered.connect(lambda: self.cameras_enabled.emit(sel, False))
        menu.exec(self.list.mapToGlobal(pos))

    def select_camera(self, i):
        for k in range(self.list.count()):
            if self.list.item(k).data(Qt.UserRole) == i:
                self.list.setCurrentRow(k); self.list.scrollToItem(self.list.item(k)); return


# ----------------------------------------------------------------------------- reference
class ReferencePane(QWidget):
    changed = Signal()                  # markers / scale bars edited -> save + redraw
    request = Signal(str)               # 'detect', 'guided', 'update', 'import', 'add_scalebar', 'remove_marker', 'remove_scalebar'
    marker_selected = Signal(str)

    MCOLS = ["Marker", "Code", "Proj.", "Err (px)", "X ref", "Y ref", "Z ref", "Acc.", "X est", "Y est", "Z est", "Error"]
    SCOLS = ["Scale bar", "Distance", "Acc.", "Estimated", "Error"]

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(2, 2, 2, 2)
        from PySide6.QtWidgets import QGridLayout, QSizePolicy
        btns = QGridLayout(); btns.setSpacing(4)
        for k, (text, key) in enumerate([("Detect markers", "detect"), ("Guided placement", "guided"),
                                         ("Import CSV…", "import"), ("Add scale bar", "add_scalebar"),
                                         ("Remove", "remove"), ("Update", "update"),
                                         ("Optimize with GCPs", "optimize_gcp")]):
            b = QPushButton(text); b.clicked.connect(lambda _=False, kk=key: self.request.emit(kk))
            b.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed); b.setMinimumWidth(60)
            btns.addWidget(b, k // 3, k % 3)
        lay.addLayout(btns)
        self.info = QLabel(); self.info.setStyleSheet("color:#8a93a3"); self.info.setWordWrap(True)
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay.addWidget(self.info)
        self.tabs = QTabWidget(); lay.addWidget(self.tabs)
        self.mt = QTableWidget(0, len(self.MCOLS)); self.mt.setHorizontalHeaderLabels(self.MCOLS)
        self.st = QTableWidget(0, len(self.SCOLS)); self.st.setHorizontalHeaderLabels(self.SCOLS)
        for t in (self.mt, self.st):
            t.setMinimumWidth(120)
            t.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
            t.setSelectionBehavior(QAbstractItemView.SelectRows)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            t.verticalHeader().setVisible(False)
        self.tabs.addTab(self.mt, "Markers"); self.tabs.addTab(self.st, "Scale bars")
        self.mt.itemChanged.connect(self._m_edit); self.st.itemChanged.connect(self._s_edit)
        self.mt.itemSelectionChanged.connect(self._m_sel)
        self.chunk = None

    def selected_markers(self):
        rows = sorted({i.row() for i in self.mt.selectedIndexes()})
        return [self.mt.item(r, 0).data(Qt.UserRole) for r in rows if self.mt.item(r, 0)]

    def selected_scalebars(self):
        return sorted({i.row() for i in self.st.selectedIndexes()})

    def _m_sel(self):
        s = self.selected_markers()
        if s:
            self.marker_selected.emit(s[0])

    def refresh(self, chunk):
        from ..export import world
        self.chunk = chunk
        T = chunk.transform is not None
        ref = chunk.stats.get("reference") or {}
        self.info.setText(("Coordinate system: %s — scale %.6g. " % (ref.get("mode"), ref.get("scale", 1)) if T else
                           "Model has arbitrary scale. Add a scale bar between two markers (known distance) or enter "
                           "reference X/Y/Z for 3+ markers, then press Update. ")
                          + "Place markers by right-clicking in a photo or with the 'Add marker' tool on the model.")
        self.mt.blockSignals(True); self.mt.setRowCount(0)
        for m in chunk.markers or []:
            r = self.mt.rowCount(); self.mt.insertRow(r)
            est = world(chunk, np.array(m["position"])[None])[0] if m.get("position") is not None else None
            refx = (m.get("reference") or {}).get("xyz")
            err = np.linalg.norm(est - np.array(refx)) if (est is not None and refx and T) else None
            vals = [m["label"], m.get("code") or "", len(m["projections"]),
                    "%.2f" % m["error_px"] if m.get("error_px") is not None else "",
                    *(["%.4f" % v for v in refx] if refx else ["", "", ""]),
                    "%.4g" % m["reference"]["accuracy"] if m.get("reference") else "",
                    *(["%.4f" % v for v in est] if est is not None else ["", "", ""]),
                    "%.4f" % err if err is not None else ""]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if c not in (0, 4, 5, 6, 7):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if c == 0:
                    it.setData(Qt.UserRole, m["label"])
                    it.setCheckState(Qt.Checked if m.get("enabled", True) else Qt.Unchecked)
                self.mt.setItem(r, c, it)
        self.mt.blockSignals(False)
        self.st.blockSignals(True); self.st.setRowCount(0)
        ms = {m["label"]: m for m in chunk.markers or []}
        for b in chunk.scalebars or []:
            r = self.st.rowCount(); self.st.insertRow(r)
            pa, pb = ms.get(b["a"], {}).get("position"), ms.get(b["b"], {}).get("position")
            est = None
            if pa is not None and pb is not None:
                est = float(np.linalg.norm(world(chunk, np.array([pa]))[0] - world(chunk, np.array([pb]))[0]))
            vals = ["%s – %s" % (b["a"], b["b"]), "%.4f" % b["distance"], "%.4g" % b.get("accuracy", 0.001),
                    "%.4f" % est if est is not None else "", "%+.4f" % (est - b["distance"]) if (est is not None and T) else ""]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c not in (1, 2):
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.st.setItem(r, c, it)
        self.st.blockSignals(False)

    def _m_edit(self, it):
        ch = self.chunk
        r, c = it.row(), it.column()
        label = self.mt.item(r, 0).data(Qt.UserRole)
        m = next((x for x in ch.markers if x["label"] == label), None)
        if m is None:
            return
        if c == 0:
            new = it.text().strip()
            m["enabled"] = it.checkState() == Qt.Checked
            if new and new != label and not any(x["label"] == new for x in ch.markers):
                for b in ch.scalebars or []:
                    b["a"] = new if b["a"] == label else b["a"]; b["b"] = new if b["b"] == label else b["b"]
                m["label"] = new
        else:
            try:
                xyz = [float(self.mt.item(r, k).text()) for k in (4, 5, 6)]
                acc = float(self.mt.item(r, 7).text()) if self.mt.item(r, 7).text() else 0.005
                m["reference"] = dict(xyz=xyz, accuracy=acc)
            except (ValueError, AttributeError):
                if all(not (self.mt.item(r, k) and self.mt.item(r, k).text().strip()) for k in (4, 5, 6)):
                    m["reference"] = None
                else:
                    return
        self.changed.emit()

    def _s_edit(self, it):
        b = self.chunk.scalebars[it.row()]
        try:
            if it.column() == 1:
                b["distance"] = float(it.text())
            elif it.column() == 2:
                b["accuracy"] = float(it.text())
        except ValueError:
            return
        self.changed.emit()


# ----------------------------------------------------------------------------- console
class ConsolePane(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setStyleSheet("font-family: Consolas, Menlo, monospace; font-size: 9pt;")

    def append_line(self, s):
        self.appendPlainText(s)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
