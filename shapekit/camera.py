"""Camera models (OpenCV convention): frame (pinhole) and fisheye (equidistant / equisolid), all with
Brown-Conrady distortion on top, like Metashape's Frame / Fisheye sensor types.

Pose convention: x_cam = R @ X_world + t   (R from Rodrigues vector rvec)
Intrinsics vector (8): f, cx, cy, k1, k2, k3, p1, p2 -- in pixels of the full-resolution image.
A 9th element, when present, is the projection model code (MODELS); without it the model is "frame".
Frame-only projects keep 8-element vectors, so their numbers are exactly as before.
"""
import numpy as np
import cv2

INTR_NAMES = ["f", "cx", "cy", "k1", "k2", "k3", "p1", "p2"]
MODELS = {"frame": 0, "fisheye_equidistant": 1, "fisheye_equisolid": 2}
MODEL_NAMES = {v: k for k, v in MODELS.items()}


def calib(intr, model="frame"):
    """Intrinsics vector for projection: 8 values for frame sensors, 9 (with the model code) for fisheye."""
    intr = np.asarray(intr, np.float64)[:8]
    code = MODELS[model] if isinstance(model, str) else int(model)
    return intr if code == 0 else np.concatenate([intr, [float(code)]])


def stack_calib(sensors):
    """(S,8) for all-frame sensor lists (unchanged behaviour), (S,9) when any sensor is fisheye."""
    codes = [MODELS[getattr(s, "model", "frame")] for s in sensors]
    I = np.array([np.asarray(s.intr, float)[:8] for s in sensors], float)
    if any(codes):
        I = np.concatenate([I, np.array(codes, float)[:, None]], 1)
    return I


def model_code(intr):
    intr = np.asarray(intr)
    return int(round(intr[8])) if intr.ndim == 1 and len(intr) > 8 else 0


def _fisheye_factor(r0, code):
    """Radial factor r_fisheye / r_pinhole for normalised pinhole radius r0 (vectorised, code per row)."""
    th = np.arctan(r0)
    safe = np.maximum(r0, 1e-12)
    fe = np.where(r0 < 1e-12, 1.0, th / safe)
    fs = np.where(r0 < 1e-12, 1.0, 2 * np.sin(th / 2) / safe)
    return np.where(code == 1, fe, np.where(code == 2, fs, 1.0))


def rodrigues_to_R(rvecs):
    """Batched Rodrigues: (N,3) -> (N,3,3)."""
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(rvecs, axis=1)
    small = theta < 1e-12
    th = np.where(small, 1.0, theta)
    k = rvecs / th[:, None]
    K = np.zeros((len(rvecs), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.sin(theta)[:, None, None]
    c = (1 - np.cos(theta))[:, None, None]
    R = np.eye(3)[None] + s * K + c * (K @ K)
    # first-order for tiny angles
    if small.any():
        rv = rvecs[small]
        Ks = np.zeros((len(rv), 3, 3))
        Ks[:, 0, 1], Ks[:, 0, 2] = -rv[:, 2], rv[:, 1]
        Ks[:, 1, 0], Ks[:, 1, 2] = rv[:, 2], -rv[:, 0]
        Ks[:, 2, 0], Ks[:, 2, 1] = -rv[:, 1], rv[:, 0]
        R[small] = np.eye(3)[None] + Ks
    return R


def R_to_rodrigues(R):
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel()


def rotate(rvecs, X):
    """Rotate points X (N,3) by per-row rvecs (N,3) (vectorized Rodrigues formula)."""
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    th = np.where(theta < 1e-12, 1.0, theta)
    k = rvecs / th
    cos, sin = np.cos(theta), np.sin(theta)
    kx = np.cross(k, X)
    kd = np.sum(k * X, axis=1, keepdims=True)
    out = X * cos + kx * sin + k * kd * (1 - cos)
    small = theta[:, 0] < 1e-12
    if small.any():
        out[small] = X[small] + np.cross(rvecs[small], X[small])
    return out


def distort(xn, intr):
    """xn: (N,2) normalized coords; intr: (N,8) or (8,). Returns pixel coords."""
    intr = np.asarray(intr)
    intr = np.broadcast_to(intr, (len(xn), len(intr))) if intr.ndim == 1 else intr
    f, cx, cy, k1, k2, k3, p1, p2 = [intr[:, i] for i in range(8)]
    x, y = xn[:, 0], xn[:, 1]
    if intr.shape[1] > 8:
        code = np.round(intr[:, 8]).astype(int)
        if code.any():
            fr = _fisheye_factor(np.hypot(x, y), code)
            x, y = x * fr, y * fr
    r2 = x * x + y * y
    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([f * xd + cx, f * yd + cy], axis=1)


def project(rvecs, tvecs, X, intr):
    """Project world points. rvecs/tvecs/intr per-observation (N,3)/(N,3)/(N,8), X (N,3).
    Returns uv (N,2) and depth z (N,)."""
    Xc = rotate(rvecs, X) + tvecs
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    xn = Xc[:, :2] / zs[:, None]
    return distort(xn, intr), z


def K_matrix(intr, scale=1.0):
    f, cx, cy = intr[0] * scale, (intr[1] + 0.5) * scale - 0.5, (intr[2] + 0.5) * scale - 0.5
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def dist_coeffs(intr):
    # OpenCV order: k1, k2, p1, p2, k3
    return np.array([intr[3], intr[4], intr[6], intr[7], intr[5]], dtype=np.float64)


def undistort_points(uv, intr):
    """Pixel coords -> normalized undistorted (pinhole) coords (N,2)."""
    if len(uv) == 0:
        return np.zeros((0, 2))
    code = model_code(intr)
    if code:
        intr = np.asarray(intr, float)
        f, cx, cy = intr[0], intr[1], intr[2]
        q = ((np.asarray(uv, np.float64).reshape(-1, 2) - [cx, cy]) / f).reshape(-1, 1, 2)
        # invert Brown in fisheye-normalised coordinates, then the fisheye mapping
        xy = cv2.undistortPoints(q, np.eye(3), dist_coeffs(intr),
                                 criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-9)).reshape(-1, 2)
        r = np.hypot(xy[:, 0], xy[:, 1])
        th = r if code == 1 else 2 * np.arcsin(np.clip(r / 2, 0, 1))
        r0 = np.tan(np.minimum(th, np.radians(89.0)))
        s = np.where(r < 1e-12, 1.0, r0 / np.maximum(r, 1e-12))
        return xy * s[:, None]
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K_matrix(intr), dist_coeffs(intr),
                              criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-9))
    return out.reshape(-1, 2)


def undistort_maps(intr, width, height, scale=1.0):
    """Remap tables producing an undistorted pinhole image at `scale` with K_new = K_matrix(intr, scale)."""
    w, h = int(round(width * scale)), int(round(height * scale))
    Knew = K_matrix(intr, scale)
    if model_code(intr):
        # direct maps: output pinhole pixel -> normalised ray -> fisheye + Brown -> source pixel
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        xn = np.stack([(xs - Knew[0, 2]) / Knew[0, 0], (ys - Knew[1, 2]) / Knew[1, 1]], -1).reshape(-1, 2)
        src = distort(xn, np.asarray(intr, float)).reshape(h, w, 2).astype(np.float32)
        return np.ascontiguousarray(src[..., 0]), np.ascontiguousarray(src[..., 1]), Knew
    m1, m2 = cv2.initUndistortRectifyMap(K_matrix(intr), dist_coeffs(intr), None, Knew, (w, h), cv2.CV_32FC1)
    return m1, m2, Knew


def camera_center(rvec, tvec):
    R = rodrigues_to_R(rvec)[0]
    return -R.T @ np.asarray(tvec)
