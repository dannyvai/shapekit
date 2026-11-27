"""View cube for the 3D model view: an orientation cube + XYZ axes in the top-right corner that turns with
the model. Click a face to look straight at it, near an edge for a 45-degree view, near a corner for a
three-quarter view.

The cube lives in its own VTK renderer (layer 1, not interactive) drawn over the main scene. Its axes are the
project's model frame: X = region R[0], Y = R[1], Z = R[2] (up), so FRONT/TOP/... match View > Front/Top/Side.
"""
import numpy as np
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor, vtkAxesActor
from vtkmodules.vtkRenderingCore import vtkRenderer

ZROT_TOP = 90       # turns the TOP/BOTTOM labels so TOP reads upright with FRONT at the bottom of the view
FACES = {(1, 0, 0): "RIGHT", (-1, 0, 0): "LEFT", (0, 1, 0): "BACK", (0, -1, 0): "FRONT",
         (0, 0, 1): "TOP", (0, 0, -1): "BOTTOM"}


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


class ViewCube:
    """Attach to a PyVista plotter: ViewCube(plotter, frame_fn) where frame_fn() returns the 3x3 model frame
    (rows = X, Y, Z-up in world coordinates) or None for the identity."""

    SIZE = 0.26            # fraction of the window height
    EDGE = 0.10            # how close to an edge/corner (cube half-size 0.5) counts as clicking it

    def __init__(self, plotter, frame_fn=None):
        self.pl = plotter
        self.frame_fn = frame_fn or (lambda: None)
        self.visible = True
        rw = plotter.ren_win
        if rw.GetNumberOfLayers() < 2:
            rw.SetNumberOfLayers(2)
        self.ren = vtkRenderer()
        self.ren.SetLayer(1)
        self.ren.InteractiveOff()                   # the main trackball never acts on the cube
        self.ren.SetViewport(0.84, 0.78, 1.0, 1.0)
        cube = vtkAnnotatedCubeActor()
        for name, txt in (("XPlus", "RIGHT"), ("XMinus", "LEFT"), ("YPlus", "BACK"), ("YMinus", "FRONT"),
                          ("ZPlus", "TOP"), ("ZMinus", "BOTTOM")):
            getattr(cube, "Set%sFaceText" % name)(txt)
        cube.SetFaceTextScale(0.15)
        cube.SetZFaceTextRotation(ZROT_TOP)
        cp = cube.GetCubeProperty()
        cp.SetColor(0.78, 0.80, 0.85); cp.SetAmbient(0.45); cp.SetDiffuse(0.55); cp.SetSpecular(0.0)
        cube.GetTextEdgesProperty().SetColor(0.25, 0.27, 0.32)
        cube.GetTextEdgesProperty().SetLineWidth(1.5)
        for get in ("GetXPlusFaceProperty", "GetXMinusFaceProperty", "GetYPlusFaceProperty",
                    "GetYMinusFaceProperty", "GetZPlusFaceProperty", "GetZMinusFaceProperty"):
            getattr(cube, get)().SetColor(0.18, 0.20, 0.25)
        self.cube = cube
        axes = vtkAxesActor()                       # X red, Y green, Z blue, from the cube's back corner
        axes.SetTotalLength(1.1, 1.1, 1.1)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.03)
        axes.SetConeRadius(0.25)
        axes.SetPosition(-0.5, -0.5, -0.5)
        self.captions = (axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D())
        for cap in self.captions:
            tp = cap.GetCaptionTextProperty()
            tp.SetFontSize(14); tp.BoldOn(); tp.ShadowOff(); tp.ItalicOff()
            cap.GetTextActor().SetTextScaleModeToNone()
        self.axes = axes
        self.ren.AddActor(cube)
        self.ren.AddActor(axes)
        self.ren.GetActiveCamera().ParallelProjectionOn()
        rw.AddRenderer(self.ren)
        self._obs = rw.AddObserver("StartEvent", lambda *a: self.sync())
        self._anim = None

    # ------------------------------------------------------------------ frame + camera sync
    def frame(self):
        R = self.frame_fn()
        if R is None:
            return np.eye(3)
        R = np.asarray(R, float)
        return R if abs(np.linalg.det(R)) > 1e-6 else np.eye(3)

    def sync(self):
        """Turn the cube like the main camera (called before every render)."""
        if not self.visible:
            return
        cam = self.pl.camera
        R = self.frame()
        d = R @ _unit(np.subtract(cam.position, cam.focal_point))       # view direction in the model frame
        up = R @ _unit(cam.up)
        c = self.ren.GetActiveCamera()
        c.SetFocalPoint(0, 0, 0)
        c.SetPosition(*(d * 6.0))
        c.SetViewUp(*up)
        c.SetParallelScale(1.12)
        for i, cap in enumerate(self.captions):                   # an axis seen end-on: its letter would sit
            cap.GetCaptionTextProperty().SetOpacity(1.0 if abs(d[i]) < 0.9 else 0.0)   # in the middle of a face
        c.SetClippingRange(0.1, 20)
        w, h = self.pl.ren_win.GetSize()
        if w > 0 and h > 0:                                        # keep it square, about SIZE of the height
            fw = min(0.4, self.SIZE * h / w)
            self.ren.SetViewport(1.0 - fw - 0.005, 1.0 - self.SIZE - 0.005, 0.995, 0.995)

    def set_visible(self, on):
        self.visible = bool(on)
        self.ren.SetDraw(1 if on else 0)
        self.pl.render()

    # ------------------------------------------------------------------ clicks
    def contains(self, x, y):
        """(x, y) in VTK display pixels (origin bottom-left)."""
        return self.visible and bool(self.ren.IsInViewport(int(x), int(y)))

    def hit(self, x, y):
        """Direction (in the model frame, components -1/0/1) for a click at display (x, y), or None."""
        self.sync()                                   # camera may have moved since the last render
        if not self.contains(x, y):
            return None
        p = self._ray_hit(x, y)
        if p is None:
            return None
        m = np.abs(p).max()
        near = np.abs(p) >= m - 1e-6                                   # the face that was hit
        near |= np.abs(p) > 0.5 - self.EDGE                            # + edges / corners close to the click
        d = np.where(near, np.sign(p), 0.0)
        return tuple(int(v) for v in d) if d.any() else None

    def _ray_hit(self, x, y):
        """Where the view ray through display (x, y) enters the cube [-0.5, 0.5]^3, or None."""
        pts = []
        for z in (0.0, 1.0):
            self.ren.SetDisplayPoint(float(x), float(y), z)
            self.ren.DisplayToWorld()
            w = self.ren.GetWorldPoint()
            pts.append(np.array(w[:3]) / (w[3] if w[3] else 1.0))
        o, d = pts[0], pts[1] - pts[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-0.5 - o) / d
            t2 = (0.5 - o) / d
        tmin = np.nanmax(np.minimum(t1, t2))
        tmax = np.nanmin(np.maximum(t1, t2))
        if not np.isfinite(tmin) or tmax < tmin or tmax < 0:
            return None
        return o + d * max(tmin, 0.0)

    def look_along(self, d_model, animate=True):
        """Look at the model from direction d_model (model frame), keeping focal point and distance."""
        cam = self.pl.camera
        R = self.frame()
        d = _unit(R.T @ _unit(d_model))                                # to world
        z = R[2]
        if abs(np.dot(d, z)) > 0.99:                                   # straight down / up: BACK at the top
            up = R[1]
        else:
            up = z - np.dot(z, d) * d
        up = _unit(up)
        f = np.array(cam.focal_point, float)
        dist = float(np.linalg.norm(np.subtract(cam.position, f))) or 1.0
        d0 = _unit(np.subtract(cam.position, f)); u0 = _unit(cam.up)

        def apply(dd, uu):
            cam.position = tuple(f + _unit(dd) * dist)
            cam.up = tuple(_unit(uu))
            self.pl.reset_camera_clipping_range()
            self.pl.render()
        if not animate:
            apply(d, up)
            return
        if np.dot(d0, d) < -0.999:                                     # opposite: turn through a side view
            mid = _unit(np.cross(d0, u0)) if np.linalg.norm(np.cross(d0, u0)) > 1e-6 else _unit(np.cross(d0, z))
            path = [(mid, u0), (d, up)]
        else:
            path = [(d, up)]
        steps = []
        a_d, a_u = d0, u0
        for b_d, b_u in path:
            for k in range(1, 9):
                t = k / 8.0
                t = t * t * (3 - 2 * t)                                # ease in/out
                steps.append((_unit((1 - t) * a_d + t * b_d), _unit((1 - t) * a_u + t * b_u)))
            a_d, a_u = b_d, b_u
        try:
            from PySide6.QtCore import QTimer
        except Exception:
            apply(d, up); return
        if self._anim is not None:
            self._anim.stop()
        timer = QTimer()
        timer.setInterval(16)
        it = iter(steps)

        def tick():
            try:
                dd, uu = next(it)
            except StopIteration:
                timer.stop(); self._anim = None
                apply(d, up)
                return
            apply(dd, uu)
        timer.timeout.connect(tick)
        self._anim = timer
        timer.start()

    def click(self, x, y):
        """Handle a click at display (x, y): True if it was on the cube (view snapped)."""
        d = self.hit(x, y)
        if d is None:
            return False
        self.look_along(d)
        return True

    def describe(self, d):
        if d in FACES:
            return FACES[d]
        parts = [FACES[tuple(int(i == k) * v for i in range(3))] for k, v in enumerate(d) if v]
        return " / ".join(parts)
