"""Photo view: zoom/pan a source photo; show / place / drag / remove marker projections; tie points overlay;
mask editing (brush and free-form, add or erase; masked areas are shown in red)."""
import numpy as np
import cv2
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPixmap, QPen, QBrush, QColor, QPainter, QAction, QFont, QImage, QPolygonF
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsSimpleTextItem,
                               QGraphicsItem, QMenu, QInputDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QCheckBox, QSpinBox, QComboBox, QPushButton)

from .. import camera as cm
from .. import masks as MS

GREEN, BLUE, RED, YELLOW = QColor("#36e26a"), QColor("#4aa3ff"), QColor("#ff4d5e"), QColor("#ffd23f")


class MarkerItem(QGraphicsEllipseItem):
    def __init__(self, view, label, x, y, r, color, movable):
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.view, self.label = view, label
        self.setPos(x, y)
        pen = QPen(color, 2); pen.setCosmetic(True)
        self.setPen(pen); self.setBrush(QBrush(Qt.NoBrush))
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        if movable:
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.setCursor(Qt.SizeAllCursor)
        self.setZValue(10)
        self.text = QGraphicsSimpleTextItem(label, self)
        self.text.setBrush(QBrush(color)); f = QFont(); f.setPointSize(9); f.setBold(True); self.text.setFont(f)
        self.text.setPos(r + 2, -r - 4)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self.flags() & QGraphicsItem.ItemIsMovable:
            p = self.pos()
            self.view.marker_moved.emit(self.label, p.x() - 0.5, p.y() - 0.5)


class PhotoView(QWidget):
    marker_moved = Signal(str, float, float)        # label, x, y (pixel-centre coords)
    marker_placed = Signal(str, float, float)       # label ('' = new), x, y
    marker_removed = Signal(str)
    mask_changed = Signal(int)

    def __init__(self, chunk, cam_index, parent=None):
        super().__init__(parent)
        self.chunk, self.cam = chunk, cam_index
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        bar = QHBoxLayout(); bar.setContentsMargins(6, 3, 6, 3)
        c = chunk.cameras[cam_index]
        from PySide6.QtWidgets import QSizePolicy
        self.info = QLabel(); self.info.setWordWrap(True); self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bar.addWidget(self.info, 1)
        self.cb_tie = QCheckBox("tie points"); self.cb_tie.toggled.connect(lambda _: self.redraw()); bar.addWidget(self.cb_tie)
        lay.addLayout(bar)
        mbar = QHBoxLayout(); mbar.setContentsMargins(6, 0, 6, 3)
        self.cb_mask = QCheckBox("Edit mask"); self.cb_mask.toggled.connect(self._mask_mode)
        self.cb_mask.setToolTip("Left drag: paint mask (ignored area) · Shift: erase · Ctrl: free-form outline "
                                "(Ctrl+Shift: erase outline) · middle drag: pan · wheel: zoom")
        self.mask_tool = QComboBox(); self.mask_tool.addItems(["Brush", "Free-form"])
        self.brush = QSpinBox(); self.brush.setRange(2, 400); self.brush.setValue(40); self.brush.setSuffix(" px")
        self.cb_show_mask = QCheckBox("show mask"); self.cb_show_mask.setChecked(True)
        self.cb_show_mask.toggled.connect(lambda _: self._update_overlay())
        b_inv = QPushButton("Invert"); b_inv.clicked.connect(self.invert_mask)
        b_clr = QPushButton("Clear"); b_clr.clicked.connect(self.clear_mask)
        for wdg in (self.cb_mask, QLabel(" tool:"), self.mask_tool, QLabel(" size:"), self.brush, self.cb_show_mask, b_inv, b_clr):
            mbar.addWidget(wdg)
        mbar.addStretch(1)
        lay.addLayout(mbar)
        self.view = _View(self)
        lay.addWidget(self.view)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.pix = self.scene.addPixmap(QPixmap(chunk.abspath(c)))
        self.pix.setZValue(0)
        sen = chunk.sensors[c.sensor]
        self.mask_keep = MS.load_mask(chunk, cam_index, (sen.width, sen.height))
        self.overlay = self.scene.addPixmap(QPixmap()); self.overlay.setZValue(2)
        self._update_overlay()
        self.items = []
        self.redraw()
        self.view.fit()

    def set_chunk(self, chunk):
        self.chunk = chunk
        self.redraw()

    def redraw(self):
        for it in self.items:
            self.scene.removeItem(it)
        self.items = []
        ch, i = self.chunk, self.cam
        c = ch.cameras[i]
        n_pin = 0
        if self.cb_tie.isChecked():
            sp = ch.load_sparse()
            if sp is not None:
                uv = sp["obs_uv"][sp["obs_cam"] == i]
                pen = QPen(QColor(255, 255, 255, 170), 1); pen.setCosmetic(True)
                for x, y in uv[:: max(1, len(uv) // 4000)]:
                    it = self.scene.addRect(QRectF(x, y, 1, 1), pen); it.setZValue(5); self.items.append(it)
        for m in ch.markers or []:
            p = m["projections"].get(str(i))
            if p is not None and not p.get("disabled"):
                col = RED if p.get("outlier") else (GREEN if p.get("pinned") or not p.get("guided") else YELLOW)
                it = MarkerItem(self, m["label"], p["xy"][0] + 0.5, p["xy"][1] + 0.5, 7, col, True)
                self.scene.addItem(it); self.items.append(it); n_pin += 1
            elif m.get("position") is not None and c.aligned:
                sen = ch.sensors[c.sensor]
                uv, z = cm.project(np.asarray(c.rvec)[None], np.asarray(c.tvec)[None], np.array(m["position"])[None], sen.calib())
                if z[0] > 0 and 0 <= uv[0, 0] < sen.width and 0 <= uv[0, 1] < sen.height:
                    it = MarkerItem(self, m["label"], uv[0, 0] + 0.5, uv[0, 1] + 0.5, 6, BLUE, True)
                    self.scene.addItem(it); self.items.append(it)
        q = "%.2f" % c.quality if c.quality is not None else "-"
        self.info.setText("%s   ·   %s   ·   quality %s   ·   %d marker projections   ·   "
                          "green = placed, yellow = guided, blue = estimated (drag to place), red = outlier"
                          % (c.label, "aligned" if c.aligned else "NOT aligned", q, n_pin))

    # ------------------------------------------------------------------ masks
    def _mask_mode(self, on):
        self.view.setDragMode(QGraphicsView.NoDrag if on else QGraphicsView.ScrollHandDrag)
        self.view.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def _keep(self):
        if self.mask_keep is None:
            sen = self.chunk.sensors[self.chunk.cameras[self.cam].sensor]
            self.mask_keep = np.ones((sen.height, sen.width), bool)
        return self.mask_keep

    def _update_overlay(self):
        if self.mask_keep is None or self.mask_keep.all() or not self.cb_show_mask.isChecked():
            self.overlay.setPixmap(QPixmap()); return
        m = ~self.mask_keep
        rgba = np.zeros(m.shape + (4,), np.uint8)
        rgba[m] = (60, 60, 255, 110)                      # BGRA for Format_ARGB32 (little endian): red, semi-transparent
        img = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_ARGB32).copy()
        self.overlay.setPixmap(QPixmap.fromImage(img))

    def paint_stroke(self, pts, erase=False, polygon=False):
        """pts: list of (x, y) in photo pixels. Brush stroke or filled outline; erase restores the pixels."""
        keep = self._keep().astype(np.uint8)
        val = 1 if erase else 0
        P = np.round(np.asarray(pts, float)).astype(np.int32)
        if polygon:
            if len(P) >= 3:
                cv2.fillPoly(keep, [P.reshape(-1, 1, 2)], val)
        else:
            r = int(self.brush.value())
            for a, b in zip(P[:-1], P[1:]) if len(P) > 1 else [(P[0], P[0])]:
                cv2.line(keep, tuple(map(int, a)), tuple(map(int, b)), val, thickness=2 * r, lineType=cv2.LINE_8)
                cv2.circle(keep, tuple(map(int, b)), r, val, -1)
        self.mask_keep = keep.astype(bool)
        self._save_mask()

    def _save_mask(self):
        MS.save_mask(self.chunk, self.cam, self.mask_keep)
        self._update_overlay()
        self.mask_changed.emit(self.cam)

    def invert_mask(self):
        self.mask_keep = ~self._keep(); self._save_mask()

    def clear_mask(self):
        self.mask_keep = None
        MS.clear_mask(self.chunk, self.cam)
        self._update_overlay(); self.mask_changed.emit(self.cam)

    def context_menu(self, scene_pos, global_pos):
        x, y = scene_pos.x() - 0.5, scene_pos.y() - 0.5
        menu = QMenu(self)
        near = None
        for it in self.items:
            if isinstance(it, MarkerItem) and (it.pos() - scene_pos).manhattanLength() < 12 / max(self.view.scale_factor(), 1e-6):
                near = it.label
        if near:
            a = menu.addAction("Remove projection of '%s'" % near)
            a.triggered.connect(lambda: self.marker_removed.emit(near))
            menu.addSeparator()
        menu.addAction("Add marker here…").triggered.connect(lambda: self.marker_placed.emit("", x, y))
        sub = menu.addMenu("Place marker")
        for m in self.chunk.markers or []:
            sub.addAction(m["label"]).triggered.connect(lambda _=False, l=m["label"]: self.marker_placed.emit(l, x, y))
        menu.exec(global_pos)


class _View(QGraphicsView):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#25282e"))

    def scale_factor(self):
        return self.transform().m11()

    def fit(self):
        self.fitInView(self.owner.pix, Qt.KeepAspectRatio)

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        s = self.scale_factor() * f
        if 0.02 < s < 64:
            self.scale(f, f)

    def mouseDoubleClickEvent(self, ev):
        if self.owner.cb_mask.isChecked():
            return
        self.fit()

    def contextMenuEvent(self, ev):
        if self.owner.cb_mask.isChecked():
            return
        self.owner.context_menu(self.mapToScene(ev.pos()), ev.globalPos())

    # mask painting / free-form outline, middle-button pan
    def mousePressEvent(self, ev):
        o = self.owner
        if ev.button() == Qt.MiddleButton:
            self._pan = ev.position(); ev.accept(); return
        if o.cb_mask.isChecked() and ev.button() == Qt.LeftButton:
            p = self.mapToScene(ev.position().toPoint())
            mods = ev.modifiers()
            self._stroke = [(p.x() - 0.5, p.y() - 0.5)]
            self._poly = bool(mods & Qt.ControlModifier) or o.mask_tool.currentText() == "Free-form"
            self._erase = bool(mods & Qt.ShiftModifier)
            pen = QPen(QColor("#00e5ff") if self._erase else QColor("#ff4d5e"), 2); pen.setCosmetic(True)
            self._path_item = self.scene().addPolygon(QPolygonF([p]), pen); self._path_item.setZValue(20)
            ev.accept(); return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if getattr(self, "_pan", None) is not None:
            d = ev.position() - self._pan; self._pan = ev.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            ev.accept(); return
        if getattr(self, "_stroke", None) is not None:
            p = self.mapToScene(ev.position().toPoint())
            self._stroke.append((p.x() - 0.5, p.y() - 0.5))
            self._path_item.setPolygon(QPolygonF([QPointF(x + 0.5, y + 0.5) for x, y in self._stroke]))
            if not self._poly and len(self._stroke) % 8 == 0:        # live brush feedback
                self.owner.paint_stroke(self._stroke[-9:], erase=self._erase)
            ev.accept(); return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self._pan = None; ev.accept(); return
        if getattr(self, "_stroke", None) is not None:
            self.owner.paint_stroke(self._stroke, erase=self._erase, polygon=self._poly)
            self.scene().removeItem(self._path_item)
            self._stroke = None; self._path_item = None
            ev.accept(); return
        super().mouseReleaseEvent(ev)
