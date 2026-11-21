"""GPU per-camera visibility + view scoring for Build Texture (same formulas as texture.build_texture)."""
import numpy as np

from .. import camera as cm
from .raster_gpu import rasterize_gpu_t


class VisibilityGPU:
    def __init__(self, V, F, cen, fn, farea):
        import torch
        self.torch = torch
        dev = self.dev = torch.device("cuda")
        self.V = torch.as_tensor(V, dtype=torch.float64, device=dev)
        self.F = torch.as_tensor(np.asarray(F, np.int64), device=dev)
        self.cen = torch.as_tensor(cen, dtype=torch.float64, device=dev)
        self.fn = torch.as_tensor(fn, dtype=torch.float64, device=dev)
        self.farea = torch.as_tensor(farea, dtype=torch.float64, device=dev)
        self.nf = len(F)

    def camera(self, cam, sen, vis_scale):
        torch = self.torch
        Kf = cm.K_matrix(sen.intr)
        Ks = cm.K_matrix(sen.intr, vis_scale)
        w, h = int(round(sen.width * vis_scale)), int(round(sen.height * vis_scale))
        R = cm.rodrigues_to_R(cam.rvec)[0]
        Rt = torch.as_tensor(R.T.copy(), device=self.dev)
        t = torch.as_tensor(np.asarray(cam.tvec, np.float64), device=self.dev)
        # raster.project_vertices
        Xv = self.V @ Rt + t
        zv = Xv[:, 2]
        zs = torch.where(zv > 1e-9, zv, torch.full_like(zv, 1e-9))
        uv = torch.stack([Xv[:, 0] / zs * Ks[0, 0] + Ks[0, 2], Xv[:, 1] / zs * Ks[1, 1] + Ks[1, 2]], 1)
        ids, zb = rasterize_gpu_t(uv.float(), zv.float(), self.F, w, h)
        Xc = self.cen @ Rt + t
        zc = Xc[:, 2]
        zm = torch.clamp_min(zc, 1e-9)
        pu = Xc[:, 0] / zm * Ks[0, 0] + Ks[0, 2]
        pv = Xc[:, 1] / zm * Ks[1, 1] + Ks[1, 2]
        m = 0.01 * w
        inb = (zc > 0) & (pu >= m) & (pu < w - m) & (pv >= m) & (pv < h - m)
        f_in = torch.nonzero(inb)[:, 0]
        zz = zb[pv[f_in].long(), pu[f_in].long()].double()
        mask = torch.zeros(self.nf, dtype=torch.bool, device=self.dev)
        mask[f_in[zc[f_in] <= zz * 1.01 + 1e-6]] = True
        hit = ids[ids >= 0].long()
        mask[hit] = True
        vis = torch.nonzero(mask)[:, 0]                     # sorted, unique (== np.union1d)
        C = torch.as_tensor(cm.camera_center(cam.rvec, cam.tvec), dtype=torch.float64, device=self.dev)
        dirv = C - self.cen[vis]
        dist = torch.sqrt((dirv * dirv).sum(1))
        cosang = (self.fn[vis] * dirv).sum(1) / torch.clamp_min(dist, 1e-12)
        good = cosang > 0.1
        vis, cosang, dist = vis[good], cosang[good], dist[good]
        area_px = self.farea[vis] * cosang * (Kf[0, 0] / dist) ** 2
        rc = torch.hypot((pu[vis] - w / 2) / (w / 2), (pv[vis] - h / 2) / (h / 2))
        q = cam.quality if cam.quality else 1.0
        score = area_px * cosang * (1 - 0.25 * torch.clamp(rc, 0, 1.4)) * (0.3 + 0.7 * q)
        return (vis.cpu().numpy(), score.cpu().numpy(), pu[vis].cpu().numpy(), pv[vis].cpu().numpy())
