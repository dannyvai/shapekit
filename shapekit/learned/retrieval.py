"""Learned global image descriptors for pair preselection / loop closure.

* "salad":  DINOv2 ViT-B/14 backbone + SALAD aggregator (Izquierdo & Civera, CVPR 2024), official weights.
            The aggregator head is re-implemented here (it is small) so the training framework
            (pytorch_lightning) is not needed; the official checkpoint is loaded into it.
* "dinov2": DINOv2 ViT-B/14, GeM-pooled patch tokens + class token (no extra weights).
Descriptors are L2-normalised; similarity = dot product. Cached per chunk in features/global_<kind>.npz.
"""
import math
import os
import numpy as np

SALAD_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _build_salad_head(torch, nn):
    def log_otp_solver(log_a, log_b, M, num_iters=20, reg=1.0):
        M = M / reg
        u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
        for _ in range(num_iters):
            u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
            v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
        return M + u.unsqueeze(2) + v.unsqueeze(1)

    def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
        b, m, n = S.size()
        S_aug = torch.empty(b, m + 1, n, dtype=S.dtype, device=S.device)
        S_aug[:, :m, :n] = S
        S_aug[:, m, :] = dustbin_score
        norm = -torch.tensor(math.log(n + m), device=S.device)
        log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
        log_a[-1] = log_a[-1] + math.log(n - m)
        log_a, log_b = log_a.expand(b, -1), log_b.expand(b, -1)
        return log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg) - norm

    class SALAD(nn.Module):
        def __init__(self, num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256):
            super().__init__()
            self.num_clusters, self.cluster_dim = num_clusters, cluster_dim
            self.token_features = nn.Sequential(nn.Linear(num_channels, 512), nn.ReLU(), nn.Linear(512, token_dim))
            self.cluster_features = nn.Sequential(nn.Conv2d(num_channels, 512, 1), nn.Identity(), nn.ReLU(),
                                                  nn.Conv2d(512, cluster_dim, 1))
            self.score = nn.Sequential(nn.Conv2d(num_channels, 512, 1), nn.Identity(), nn.ReLU(),
                                       nn.Conv2d(512, num_clusters, 1))
            self.dust_bin = nn.Parameter(torch.tensor(1.))

        def forward(self, x):
            x, t = x
            f = self.cluster_features(x).flatten(2)
            p = self.score(x).flatten(2)
            t = self.token_features(t)
            p = get_matching_probs(p, self.dust_bin, 3)
            p = torch.exp(p)[:, :-1, :]
            p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
            f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
            f = torch.cat([nn.functional.normalize(t, p=2, dim=-1),
                           nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)], dim=-1)
            return nn.functional.normalize(f, p=2, dim=-1)
    return SALAD


class GlobalDescriptor:
    def __init__(self, kind="salad"):
        import torch
        import torch.nn as nn
        self.torch, self.kind = torch, kind
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
        if os.path.isfile(os.path.join(local, "hubconf.py")):
            # cached / bundled (portable build): load offline from the local copy of the repo
            dino = torch.hub.load(local, "dinov2_vitb14", source="local", verbose=False)
        else:
            dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        self.head = None
        if kind == "salad":
            sd = torch.hub.load_state_dict_from_url(SALAD_URL, map_location="cpu", file_name="dino_salad.ckpt")
            sd = sd.get("state_dict", sd)
            bb = {k[len("backbone.model."):]: v for k, v in sd.items() if k.startswith("backbone.model.")}
            ag = {k[len("aggregator."):]: v for k, v in sd.items() if k.startswith("aggregator.")}
            missing = dino.load_state_dict(bb, strict=False)
            if missing.missing_keys:
                raise RuntimeError("SALAD backbone weights incomplete: %s" % missing.missing_keys[:5])
            self.head = _build_salad_head(torch, nn)()
            self.head.load_state_dict(ag, strict=True)
            self.head = self.head.eval().to(self.dev)
        self.dino = dino.eval().to(self.dev)

    def _prep(self, img_bgr, height=322):
        import cv2
        h, w = img_bgr.shape[:2]
        H = height
        W = max(14, int(round(w * H / h / 14)) * 14)
        im = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
        return ((im - MEAN) / STD).transpose(2, 0, 1)

    def describe(self, images, batch=16):
        """images: list of BGR uint8 arrays (same size). Returns (N, D) float32 L2-normalised."""
        torch = self.torch
        out = []
        for b0 in range(0, len(images), batch):
            x = torch.from_numpy(np.stack([self._prep(im) for im in images[b0:b0 + batch]])).to(self.dev)
            with torch.inference_mode(), torch.autocast("cuda", enabled=self.dev.type == "cuda", dtype=torch.float16):
                B, _, H, W = x.shape
                t = self.dino.prepare_tokens_with_masks(x)
                for blk in self.dino.blocks:
                    t = blk(t)
                t = self.dino.norm(t)
                cls, patches = t[:, 0], t[:, 1:]
                f = patches.reshape(B, H // 14, W // 14, -1).permute(0, 3, 1, 2).float()
                if self.head is not None:
                    with torch.autocast("cuda", enabled=False):       # Sinkhorn in fp32
                        d = self.head((f, cls.float()))
                else:   # GeM (p=3) over patches + class token
                    g = f.clamp_min(1e-6).pow(3).mean((2, 3)).pow(1 / 3)
                    d = torch.cat([torch.nn.functional.normalize(g, dim=1),
                                   torch.nn.functional.normalize(cls.float(), dim=1)], 1)
                    d = torch.nn.functional.normalize(d, dim=1)
            out.append(d.float().cpu().numpy())
        return np.concatenate(out)


def global_descriptors(chunk, kind="salad", force=False):
    """Compute (or load cached) global descriptors for all cameras of the chunk."""
    from ..util import log, read_image
    p = os.path.join(chunk.dir, "features", "global_%s.npz" % kind)
    labels = np.array([c.label for c in chunk.cameras])
    if os.path.exists(p) and not force:
        d = np.load(p)
        if len(d["labels"]) == len(labels) and (d["labels"] == labels).all():
            return d["desc"]
    import time
    t0 = time.time()
    g = GlobalDescriptor(kind)
    descs = []
    B = 16
    for b0 in range(0, len(chunk.cameras), B):
        ims = [read_image(chunk.abspath(c), scale=0.5) for c in chunk.cameras[b0:b0 + B]]
        shapes = {im.shape for im in ims}
        if len(shapes) == 1:
            descs.append(g.describe(ims, B))
        else:
            descs += [g.describe([im]) for im in ims]
    D = np.concatenate(descs).astype(np.float32)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, desc=D, labels=labels)
    log("  global descriptors (%s): %d images, %d-D, %.1fs" % (kind, len(D), D.shape[1], time.time() - t0))
    return D
