"""Shared GUI helpers: background tasks, log bridge, parameter dialogs."""
import traceback

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox,
                               QCheckBox, QLineEdit, QVBoxLayout, QLabel)


class LogBridge(QObject):
    line = Signal(str)


class Task(QThread):
    """Runs fn(*a, **k) off the UI thread. finished_with(result, error_text_or_None)."""
    finished_with = Signal(object, object)

    def __init__(self, fn, *a, **k):
        super().__init__()
        self.fn, self.a, self.k = fn, a, k

    def run(self):
        try:
            r = self.fn(*self.a, **self.k)
            self.finished_with.emit(r, None)
        except Exception:
            self.finished_with.emit(None, traceback.format_exc())


class ParamDialog(QDialog):
    """fields: list of (key, label, kind, default, options)
       kind: 'choice' (options = list), 'int' (options=(min,max)), 'float' (options=(min,max,decimals)), 'bool', 'text'"""

    def __init__(self, title, fields, parent=None, note=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.widgets = {}
        lay = QVBoxLayout(self)
        if note:
            lb = QLabel(note); lb.setWordWrap(True); lb.setStyleSheet("color:#667")
            lay.addWidget(lb)
        form = QFormLayout()
        lay.addLayout(form)
        for key, label, kind, default, opts in fields:
            if kind == "choice":
                w = QComboBox(); w.addItems([str(o) for o in opts]); w.setCurrentText(str(default))
            elif kind == "int":
                w = QSpinBox(); w.setRange(*opts); w.setValue(int(default))
            elif kind == "float":
                w = QDoubleSpinBox(); w.setRange(opts[0], opts[1]); w.setDecimals(opts[2] if len(opts) > 2 else 3)
                w.setValue(float(default))
            elif kind == "bool":
                w = QCheckBox(); w.setChecked(bool(default))
            else:
                w = QLineEdit(str(default))
            self.widgets[key] = (kind, w)
            form.addRow(label, w)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def values(self):
        out = {}
        for k, (kind, w) in self.widgets.items():
            if kind == "choice":
                out[k] = w.currentText()
            elif kind in ("int", "float"):
                out[k] = w.value()
            elif kind == "bool":
                out[k] = w.isChecked()
            else:
                out[k] = w.text()
        return out

    @staticmethod
    def ask(title, fields, parent=None, note=None):
        d = ParamDialog(title, fields, parent, note)
        return d.values() if d.exec() == QDialog.Accepted else None
