"""Generate Report: self-contained HTML processing report with an interactive 3D viewer."""
import base64
import html
import os
import tempfile
import time
import numpy as np
import cv2

from . import camera as cm
from .export import export_model, viewer_transform
from .util import log, read_image

CSS = """
:root{--ink:#1d2330;--mut:#6b7385;--line:#e3e6ec;--bg:#f6f7f9;--acc:#2f6fdf}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
header{padding:28px 40px 18px;background:#fff;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:22px}h2{font-size:16px;margin:32px 0 12px}.sub{color:var(--mut)}
main{padding:0 40px 60px;max-width:1300px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;margin-top:20px}
.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card b{display:block;font-size:20px;font-variant-numeric:tabular-nums}.card span{color:var(--mut);font-size:12px}
table{border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;width:100%}
td,th{padding:7px 12px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums}
th{background:#fafbfc;font-weight:600;color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
#viewer{width:100%;height:620px;background:#20242c;border-radius:8px;position:relative}
#viewer .hint{position:absolute;left:12px;bottom:10px;color:#aab;font-size:12px}
img.fig{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
"""

VIEWER_JS = """
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const el=document.getElementById('viewer');
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(el.clientWidth,el.clientHeight);renderer.outputColorSpace=THREE.SRGBColorSpace;el.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x20242c);
const cam=new THREE.PerspectiveCamera(50,el.clientWidth/el.clientHeight,0.01,5000);
const ctl=new OrbitControls(cam,renderer.domElement);ctl.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,2.5));
const b64=document.getElementById('glb').textContent.trim();
const bin=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
new GLTFLoader().parse(bin.buffer,'',g=>{scene.add(g.scene);
  const box=new THREE.Box3().setFromObject(g.scene);const s=box.getSize(new THREE.Vector3()).length();
  const c=box.getCenter(new THREE.Vector3());ctl.target.copy(c);cam.position.set(c.x+s*0.35,c.y+s*0.55,c.z+s*0.6);
  cam.near=s/1000;cam.far=s*20;cam.updateProjectionMatrix();
  const cams=JSON.parse(document.getElementById('cams').textContent);
  const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(cams.flat(),3));
  const pts=new THREE.Points(geo,new THREE.PointsMaterial({color:0xff5a4e,size:s/150}));pts.visible=false;scene.add(pts);
  document.getElementById('togcams').onclick=()=>{pts.visible=!pts.visible};
});
addEventListener('resize',()=>{renderer.setSize(el.clientWidth,el.clientHeight);cam.aspect=el.clientWidth/el.clientHeight;cam.updateProjectionMatrix()});
(function loop(){requestAnimationFrame(loop);ctl.update();renderer.render(scene,cam)})();
"""


def _img_b64(img, ext=".jpg", q=85):
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, q] if ext == ".jpg" else [])
    return "data:image/%s;base64,%s" % ("jpeg" if ext == ".jpg" else "png", base64.b64encode(buf).decode())


def overview_plot(chunk, size=900):
    sp = chunk.load_sparse()
    c, M = viewer_transform(chunk)
    P = (sp["points"] - c) @ M.T
    cams = np.array([(chunk.cameras[i].center - c) @ M.T for i in chunk.aligned_cameras()])
    xs, zs = P[:, 0], P[:, 2]
    allx = np.concatenate([np.percentile(xs, [1, 99]), cams[:, 0]]); allz = np.concatenate([np.percentile(zs, [1, 99]), cams[:, 2]])
    lo = np.array([allx.min(), allz.min()]); hi = np.array([allx.max(), allz.max()])
    span = (hi - lo).max() * 1.08
    mid = (hi + lo) / 2
    img = np.full((size, size, 3), 255, np.uint8)

    def px(x, z):
        return (((x - mid[0]) / span + 0.5) * size).astype(int), (((z - mid[1]) / span + 0.5) * size).astype(int)

    u, v = px(xs, zs)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    order = np.argsort(P[ok, 1])
    img[v[ok][order], u[ok][order]] = sp["colors"][ok][order][:, ::-1]
    cu, cv_ = px(cams[:, 0], cams[:, 2])
    for k in range(len(cu) - 1):
        cv2.line(img, (cu[k], cv_[k]), (cu[k + 1], cv_[k + 1]), (60, 60, 230), 1, cv2.LINE_AA)
    for a, b in zip(cu, cv_):
        cv2.circle(img, (a, b), 4, (40, 40, 220), -1, cv2.LINE_AA)
    # scale-free grid note
    cv2.putText(img, "top view: tie points + camera positions", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def comparison_figs(chunk, n=3, scale=0.4):
    from . import raster
    from .mesh import load_mesh
    uvp = chunk.path("model", "uv.npz")
    if not os.path.exists(uvp):
        return []
    V, F, _ = load_mesh(chunk)
    d = np.load(uvp)
    atlas = cv2.imread(chunk.path("model", "model_texture.jpg"))
    al = chunk.aligned_cameras()
    out = []
    for i in [al[int(k)] for k in np.linspace(0, len(al) - 1, n + 2)[1:-1]]:
        cam = chunk.cameras[i]; sen = chunk.sensors[cam.sensor]
        # render at exactly the undistorted photo's size and intrinsics (int() vs round() differed by a pixel
        # for e.g. 4872 x 0.4 = 1948.8, and the side-by-side failed)
        m1, m2, K = cm.undistort_maps(sen.calib(), sen.width, sen.height, scale)
        ref = cv2.remap(read_image(chunk.abspath(cam)), m1, m2, cv2.INTER_AREA)
        h, w = ref.shape[:2]
        r, _ = raster.render_textured(V, F, d["uv"], d["face_uv"], atlas, cam.rvec, cam.tvec, K, w, h, bg=(44, 36, 32))
        if r.shape[:2] != ref.shape[:2]:
            r = cv2.resize(r, (w, h), interpolation=cv2.INTER_AREA)
        out.append((cam.label, np.hstack([ref, r])))
    return out


def per_camera_error(chunk):
    sp = chunk.load_sparse()
    e = np.zeros(len(chunk.cameras)); n = np.zeros(len(chunk.cameras))
    np.add.at(e, sp["obs_cam"], sp["obs_err"].astype(float) ** 2)
    np.add.at(n, sp["obs_cam"], 1)
    return np.sqrt(e / np.maximum(n, 1)), n


def bar_svg(vals, labels, w=1200, h=140, unit="px"):
    vals = np.asarray(vals, float)
    mx = max(vals.max(), 1e-9)
    bw = w / max(len(vals), 1)
    bars = "".join('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#2f6fdf"><title>%s: %.2f %s</title></rect>'
                   % (k * bw + 0.5, h - 18 - v / mx * (h - 30), max(bw - 1, 0.5), v / mx * (h - 30), html.escape(l), v, unit)
                   for k, (v, l) in enumerate(zip(vals, labels)))
    return ('<svg viewBox="0 0 %d %d" width="100%%" style="background:#fff;border:1px solid #e3e6ec;border-radius:8px">%s'
            '<text x="6" y="%d" font-size="11" fill="#6b7385">max %.2f %s</text></svg>' % (w, h, bars, h - 4, mx, unit))


def generate_report(chunk, path=None, viewer_max_faces=600000, viewer_texture=4096):
    t0 = time.time()
    path = path or os.path.join(chunk.project.dir, "report.html")
    st = chunk.stats
    a = st.get("align", {})
    cards = [("Cameras aligned", "%s / %s" % (a.get("aligned", "-"), a.get("cameras", len(chunk.cameras)))),
             ("Tie points", "{:,}".format(a.get("tie_points", 0))),
             ("Reprojection RMS", "%.2f px" % a.get("rms_reprojection_px", float("nan"))),
             ("Dense points", "{:,}".format(st.get("point_cloud", {}).get("points", 0))),
             ("Mesh faces", "{:,}".format(st.get("mesh", {}).get("faces", 0))),
             ("Depth maps", "%s, %s filter, %s" % (st.get("depth_maps", {}).get("quality", "-"),
                                                   st.get("depth_maps", {}).get("filter", "-"),
                                                   st.get("depth_maps", {}).get("method", "planesweep"))),
             ("Texture", "%s px, %s" % (st.get("texture", {}).get("size", "-"), st.get("texture", {}).get("blending", "mosaic")))]
    sens_rows = "".join("<tr><td>%s</td><td>%dx%d</td>%s</tr>" % (
        html.escape(s.label + ("" if getattr(s, "model", "frame") == "frame" else " (%s)" % s.model)), s.width, s.height,
        "".join("<td>%.5g</td>" % v for v in s.intr[:8])) for s in chunk.sensors)
    log_rows = "".join("<tr><td>%s</td><td>%s</td><td>%.1f s</td><td>%s</td></tr>" % (
        html.escape(l["step"]), l["time"], l["seconds"],
        html.escape(", ".join("%s=%s" % (k, (round(v, 3) if isinstance(v, float) else v)) for k, v in l.items()
                              if k not in ("step", "time", "seconds")))) for l in chunk.log)
    err, nobs = per_camera_error(chunk) if chunk.load_sparse() is not None else ([], [])
    al = chunk.aligned_cameras()
    sources = sorted({c.source or "photos" for c in chunk.cameras})
    viewer = ""
    if os.path.exists(chunk.path("mesh.ply")):
        with tempfile.TemporaryDirectory() as td:
            g = os.path.join(td, "v.glb")
            export_model(chunk, g, max_faces=viewer_max_faces, texture_size=viewer_texture, local_frame=True)
            glb = base64.b64encode(open(g, "rb").read()).decode()
        c, M = viewer_transform(chunk)
        camspos = [[round(float(x), 4) for x in (chunk.cameras[i].center - c) @ M.T] for i in al]
        import json
        viewer = ('<h2>3D model</h2><div id="viewer"><div class="hint">drag: rotate · right-drag: pan · wheel: zoom · '
                  '<a href="#" id="togcams" style="color:#ccd">toggle cameras</a></div></div>'
                  '<script id="glb" type="application/octet-stream">%s</script>'
                  '<script id="cams" type="application/json">%s</script>' % (glb, json.dumps(camspos)))
    ref_html = ""
    if chunk.markers:
        from .export import world
        T_set = getattr(chunk, "transform", None) is not None
        rows = ""
        for m in chunk.markers:
            pos = "" if m.get("position") is None else "%.4f, %.4f, %.4f" % tuple(world(chunk, np.array(m["position"])[None])[0])
            ref = (m.get("reference") or {}).get("xyz")
            merr = "" if not ref or m.get("position") is None else "%.4f" % np.linalg.norm(
                world(chunk, np.array(m["position"])[None])[0] - np.array(ref))
            rows += "<tr><td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                html.escape(m["label"]), m.get("code") or "", len(m["projections"]),
                "%.2f" % m["error_px"] if m.get("error_px") else "", pos, "" if not ref else "%.4f, %.4f, %.4f" % tuple(ref), merr)
        sb = ""
        for b in chunk.scalebars or []:
            ms = {m["label"]: m for m in chunk.markers}
            if ms.get(b["a"], {}).get("position") is None or ms.get(b["b"], {}).get("position") is None:
                continue
            d = np.linalg.norm(world(chunk, np.array([ms[b["a"]]["position"]]))[0] - world(chunk, np.array([ms[b["b"]]["position"]]))[0])
            sb += "<tr><td>%s – %s</td><td>%.4f</td><td>%.4f</td><td>%+.4f</td></tr>" % (
                html.escape(b["a"]), html.escape(b["b"]), b["distance"], d, d - b["distance"])
        st_ref = chunk.stats.get("reference", {})
        ref_html = ('<h2>Markers &amp; reference</h2><p class="sub">%s</p>'
                    '<table><tr><th>Marker</th><th>Code</th><th>Projections</th><th>Reproj. px</th><th>Estimated (world)</th>'
                    '<th>Reference</th><th>Error</th></tr>%s</table>' % (
                        ("Coordinate system: %s, scale %.6g" % (st_ref.get("mode"), st_ref.get("scale", 1))) if T_set
                        else "No scale bars or reference coordinates: the model has arbitrary scale.", rows))
        if sb:
            ref_html += ('<h2>Scale bars</h2><table><tr><th>Scale bar</th><th>Distance</th><th>Estimated</th>'
                         '<th>Error</th></tr>%s</table>' % sb)
    figs = comparison_figs(chunk)
    fig_html = "".join('<figure style="margin:0 0 12px"><img class="fig" src="%s"><figcaption class="sub">%s — source frame (left) vs. rendered model (right)</figcaption></figure>'
                       % (_img_b64(img), html.escape(lbl)) for lbl, img in figs)
    doc = """<!doctype html><html><head><meta charset="utf-8"><title>Processing report</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>%s</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
</head><body><header><h1>Processing report — %s</h1><div class="sub">%s · generated %s · sources: %s</div>
<div class="cards">%s</div></header><main>
%s
<h2>Survey overview</h2><img class="fig" src="%s" style="max-width:700px">
<h2>Reprojection error per camera</h2>%s
<h2>Camera calibration</h2><table><tr><th>Sensor</th><th>Resolution</th>%s</tr>%s</table>
%s
<h2>Model vs. photos</h2>%s
<h2>Processing log</h2><table><tr><th>Step</th><th>Finished</th><th>Duration</th><th>Parameters / result</th></tr>%s</table>
</main><script type="module">%s</script></body></html>""" % (
        CSS, html.escape(chunk.label), html.escape(chunk.project.dir), time.strftime("%Y-%m-%d %H:%M"),
        html.escape(", ".join(sources)),
        "".join('<div class="card"><b>%s</b><span>%s</span></div>' % (v, k) for k, v in cards),
        viewer, _img_b64(overview_plot(chunk), ".png") if a else "",
        bar_svg(np.asarray(err)[al], [chunk.cameras[i].label for i in al]) if len(al) else "",
        "".join("<th>%s</th>" % n for n in cm.INTR_NAMES), sens_rows, ref_html, fig_html, log_rows,
        VIEWER_JS if viewer else "")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    log("Report -> %s (%.1f MB, %.0fs)" % (path, os.path.getsize(path) / 1e6, time.time() - t0))
    return path
