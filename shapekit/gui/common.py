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
        self.defaults = {f[0]: f[3] for f in fields}
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
        b_def = bb.addButton("Defaults", QDialogButtonBox.ResetRole)
        b_def.setToolTip("Put back the original values")
        b_def.clicked.connect(lambda: self.set_values(self.defaults))
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

    def set_values(self, vals):
        for k, v in (vals or {}).items():
            if k not in self.widgets:
                continue
            kind, w = self.widgets[k]
            try:
                if kind == "choice":
                    w.setCurrentText(str(v))
                elif kind == "int":
                    w.setValue(int(v))
                elif kind == "float":
                    w.setValue(float(v))
                elif kind == "bool":
                    w.setChecked(bool(v))
                else:
                    w.setText(str(v))
            except (TypeError, ValueError):
                pass

    last = None           # (title, values) of the most recently accepted dialog -> crash-recovery record
    prefill = None        # {title: values}: open these dialogs with these values (resuming a crashed task)
    memory_path = None    # <project>/dialog_settings.json: last values per dialog, per project (set by the app)
    # options that throw work away are never remembered: they always open at their (safe) default
    NEVER_REMEMBER = {"reset", "reset_features", "force", "adjust"}

    @classmethod
    def _memory(cls):
        import json
        if not cls.memory_path:
            return {}
        try:
            with open(cls.memory_path, encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    @classmethod
    def _remember(cls, title, values):
        import json, os
        if not cls.memory_path:
            return
        d = cls._memory()
        d[title] = {k: v for k, v in values.items() if k not in cls.NEVER_REMEMBER}
        try:
            tmp = cls.memory_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
            os.replace(tmp, cls.memory_path)
        except OSError:
            pass

    @staticmethod
    def ask(title, fields, parent=None, note=None):
        d = ParamDialog(title, fields, parent, note)
        if ParamDialog.prefill and title in ParamDialog.prefill:
            d.set_values(ParamDialog.prefill[title])
            d.setWindowTitle(title + " — continue after crash")
        else:
            saved = ParamDialog._memory().get(title)
            if saved:                                   # the values last used for this dialog in this project
                d.set_values({k: v for k, v in saved.items() if k not in ParamDialog.NEVER_REMEMBER})
                d.setWindowTitle(title + " — last used settings")
        if d.exec() != QDialog.Accepted:
            return None
        ParamDialog.last = (title, d.values())
        ParamDialog._remember(title, ParamDialog.last[1])
        return ParamDialog.last[1]
