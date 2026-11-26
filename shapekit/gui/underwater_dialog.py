"""Tools > Underwater Colour...: white balance from a reference patch, or distance-aware water removal."""
import numpy as np
from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QPushButton, QLabel,
                               QDoubleSpinBox, QRubberBand, QColorDialog, QDialogButtonBox, QMessageBox, QFrame)

from .. import underwater as UW


def to_pixmap(bgr, max_w=None, max_h=None):
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    img = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()
    pm = QPixmap.fromImage(img)
    if max_w and max_h:
        pm = pm.scaled(QSize(max_w, max_h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm


class _PickLabel(QLabel):
    """Shows an image; drag a rectangle with the left mouse button."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.band = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = None
        self.rect_px = None
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.origin = e.position().toPoint()
            self.band.setGeometry(QRect(self.origin, QSize()))
            self.band.show()

    def mouseMoveEvent(self, e):
        if self.origin is not None:
            self.band.setGeometry(QRect(self.origin, e.position().toPoint()).normalized())

    def mouseReleaseEvent(self, e):
        if self.origin is not None:
            r = QRect(self.origin, e.position().toPoint()).normalized()
            self.origin = None
            self.rect_px = r if r.width() > 3 and r.height() > 3 else None
            if self.rect_px is None:
                self.band.hide()


class PatchPicker(QDialog):
    """Pick the reference patch (sandbag, grey card, white slate...) on one photo."""

    def __init__(self, chunk, cam_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pick reference patch")
        self.chunk = chunk
        self.cams = chunk.aligned_cameras() or list(range(len(chunk.cameras)))
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Photo:"))
        self.combo = QComboBox()
        for i in self.cams:
            self.combo.addItem(chunk.cameras[i].label, i)
        top.addWidget(self.combo, 1)
        prev, nxt = QPushButton("◀"), QPushButton("▶")
        prev.clicked.connect(lambda: self.combo.setCurrentIndex(max(0, self.combo.currentIndex() - 1)))
        nxt.clicked.connect(lambda: self.combo.setCurrentIndex(min(self.combo.count() - 1, self.combo.currentIndex() + 1)))
        top.addWidget(prev); top.addWidget(nxt)
        lay.addLayout(top)
        hint = QLabel("Drag a box over an area that should be neutral (or of the colour you set): a sandbag, grey card, "
                      "white slate... Avoid shadows, highlights and fish.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#667")
        lay.addWidget(hint)
        self.view = _PickLabel()
        lay.addWidget(self.view, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.combo.currentIndexChanged.connect(self._load)
        k = self.cams.index(cam_index) if cam_index in self.cams else 0
        self.combo.setCurrentIndex(k)
        self._load()
        self.result = None

    def _load(self):
        i = self.combo.currentData()
        if i is None:
            return
        und, _ = UW.undistorted(self.chunk, i, 1.0)
        self.full_wh = (und.shape[1], und.shape[0])
        pm = to_pixmap(und, 1300, 800)
        self.scale = pm.width() / und.shape[1]
        self.view.setPixmap(pm)
        self.view.setFixedSize(pm.size())
        self.view.band.hide(); self.view.rect_px = None

    def _ok(self):
        r = self.view.rect_px
        if r is None:
            QMessageBox.information(self, "Reference patch", "Drag a box over the reference area first."); return
        s = self.scale
        i = self.combo.currentData()
        self.result = dict(camera=int(i), label=self.chunk.cameras[i].label,
                           rect=[r.left() / s, r.top() / s, (r.right() + 1) / s, (r.bottom() + 1) / s])
        self.accept()


class UnderwaterDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.chunk = main.chunk
        self.setWindowTitle("Underwater Colour")
        self.s = UW.load_settings(self.chunk)
        lay = QVBoxLayout(self)
        note = QLabel("Corrects the colours of the photos used by <b>Build Texture</b> (the photos themselves are not "
                      "changed). <b>White balance</b>: one correction for all photos, right at the reference patch's "
                      "distance. <b>Distance-aware</b>: removes the water's blue-green haze and the red loss per pixel, "
                      "using the distance to the mesh (needs Build Mesh); the reference patch then sets the white point.")
        note.setWordWrap(True); note.setStyleSheet("color:#556")
        lay.addWidget(note)
        form = QFormLayout(); lay.addLayout(form)
        self.mode = QComboBox()
        for m in UW.MODES:
            self.mode.addItem(UW.MODE_LABELS[m], m)
        self.mode.setCurrentIndex(UW.MODES.index(self.s["mode"]))
        form.addRow("Correction", self.mode)
        row = QHBoxLayout()
        self.patch_lb = QLabel()
        b_pick, b_clear = QPushButton("Pick on photo…"), QPushButton("Clear")
        b_pick.clicked.connect(self.pick); b_clear.clicked.connect(lambda: self._set_patch(None))
        row.addWidget(self.patch_lb, 1); row.addWidget(b_pick); row.addWidget(b_clear)
        form.addRow("Reference patch", row)
        row2 = QHBoxLayout()
        self.ref = QComboBox(); self.ref.addItems(["Neutral (grey / white)", "Known colour"])
        self.swatch = QPushButton(); self.swatch.setFixedWidth(70)
        self.swatch.clicked.connect(self.pick_colour)
        row2.addWidget(self.ref, 1); row2.addWidget(self.swatch)
        form.addRow("Patch true colour", row2)
        self.strength = QDoubleSpinBox(); self.strength.setRange(0.0, 1.5); self.strength.setSingleStep(0.05)
        self.strength.setDecimals(2); self.strength.setValue(float(self.s.get("strength", 1.0)))
        form.addRow("Strength (distance-aware)", self.strength)
        self.info = QLabel(); self.info.setWordWrap(True); self.info.setStyleSheet("color:#446")
        lay.addWidget(self.info)
        prev = QHBoxLayout()
        self.before, self.after = QLabel("before"), QLabel("after")
        for lb in (self.before, self.after):
            lb.setFrameShape(QFrame.StyledPanel); lb.setAlignment(Qt.AlignCenter); lb.setMinimumSize(420, 280)
            prev.addWidget(lb)
        lay.addLayout(prev)
        btns = QHBoxLayout()
        self.b_est = QPushButton("Estimate && Preview")
        self.b_est.clicked.connect(self.estimate)
        b_save, b_close = QPushButton("Save"), QPushButton("Close")
        b_save.clicked.connect(self.save); b_close.clicked.connect(self.close)
        btns.addWidget(self.b_est); btns.addStretch(1); btns.addWidget(b_save); btns.addWidget(b_close)
        lay.addLayout(btns)
        self.ref_rgb = self.s.get("ref_rgb") or [200, 190, 160]
        self.ref.setCurrentIndex(1 if self.s.get("ref_rgb") else 0)
        self.ref.currentIndexChanged.connect(self._refresh)
        self.mode.currentIndexChanged.connect(self._refresh)
        self._set_patch(self.s.get("patch"))
        self._show_params(self.s.get("params"))

    # ---------------------------------------------------------------- ui state
    def _refresh(self):
        self.swatch.setEnabled(self.ref.currentIndex() == 1)
        self.swatch.setStyleSheet("background: rgb(%d,%d,%d)" % tuple(self.ref_rgb))
        self.strength.setEnabled(self.mode.currentData() == "seathru")
        self.b_est.setEnabled(self.mode.currentData() != "off")

    def _set_patch(self, p):
        self.patch = p
        if p:
            x0, y0, x1, y1 = p["rect"]
            self.patch_lb.setText("%s, %dx%d px" % (p.get("label", p["camera"]), x1 - x0, y1 - y0))
        else:
            self.patch_lb.setText("none (grey-world: assumes the scene is grey on average)")
        self._refresh()

    def _show_params(self, p):
        if not p:
            self.info.setText("Not estimated yet: press Estimate && Preview (or it is estimated in Build Texture).")
            return
        if p.get("kind") == "wb":
            self.info.setText("White balance gains R %.2f G %.2f B %.2f (from %s), %s"
                              % (*p["gains"], p.get("white"), p.get("estimated", "")))
        else:
            self.info.setText("Water: attenuation R %.3g G %.3g B %.3g per unit, backscatter R %.3f G %.3f B %.3f "
                              "(%s); white point from %s. Median distance %.3g units. %s"
                              % (*p["beta"], *p["Binf"], p.get("attenuation_from"), p.get("white"),
                                 p.get("median_range", 0), p.get("estimated", "")))

    def current(self):
        s = dict(self.s)
        s.update(mode=self.mode.currentData(), patch=self.patch, strength=float(self.strength.value()),
                 ref_rgb=list(map(int, self.ref_rgb)) if self.ref.currentIndex() == 1 else None)
        return s

    # ---------------------------------------------------------------- actions
    def pick(self):
        sel = self.main.photos.selected() if hasattr(self.main, "photos") else []
        start = (self.patch or {}).get("camera", sel[0] if sel else None)
        d = PatchPicker(self.chunk, start, self)
        if d.exec() == QDialog.Accepted and d.result:
            self._set_patch(d.result)

    def pick_colour(self):
        c = QColorDialog.getColor(QColor(*self.ref_rgb), self, "True colour of the reference patch (in air)")
        if c.isValid():
            self.ref_rgb = [c.red(), c.green(), c.blue()]
            self._refresh()

    def save(self):
        s = self.current()
        if s.get("params") and s["params"].get("key") != UW._settings_key(s):
            s["params"] = None                       # settings changed -> re-estimated at the next Build Texture
        UW.save_settings(self.chunk, s)
        self.s = s
        self.main.status("Underwater colour saved (%s) — rebuild the texture to apply" % UW.MODE_LABELS[s["mode"]])
        self._show_params(s.get("params"))

    def estimate(self):
        s = self.current()
        if s["mode"] == "seathru" and not self.chunk.stats.get("mesh"):
            QMessageBox.information(self, "Underwater Colour", "The distance-aware correction needs a mesh: run Build Mesh first.")
            return
        sel = self.main.photos.selected() if hasattr(self.main, "photos") else []
        cam = (s.get("patch") or {}).get("camera", sel[0] if sel else (self.chunk.aligned_cameras() or [0])[0])
        chunk = self.chunk

        def job():
            s["params"] = UW.estimate(chunk, s)
            UW.save_settings(chunk, s)
            return s, UW.preview(chunk, cam, s, scale=0.35)

        def done(res):
            if not res:
                return
            self.s, (bef, aft) = res
            self._show_params(self.s["params"])
            self.before.setPixmap(to_pixmap(bef, 520, 360)); self.after.setPixmap(to_pixmap(aft, 520, 360))
            self.main.status("Underwater colour estimated and saved — rebuild the texture to apply")
        self.main.run("Estimate underwater colour", job, done=done)
