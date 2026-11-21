"""Depth-map GPU kernel: legacy batch size vs memory-sized batch (must be bit-identical)."""
import os, sys, time
import numpy as np


def main(src="projects/gpu_test", n=12):
    from shapekit.project import Project
    from shapekit import mvs
    from shapekit.learned.mvs_gpu import compute_depth_gpu
    ch = Project.open(src).chunk()
    nb = mvs.select_neighbors(ch, 5)
    cams = [i for i in ch.aligned_cameras() if nb.get(i)][:n]
    views = {}
    V = lambda i: views.setdefault(i, mvs.View(ch, i, 0.5))
    res = {}
    for mode in ("legacy", "fast"):
        if mode == "legacy": os.environ["SHAPEKIT_LEGACY"] = "1"
        else: os.environ.pop("SHAPEKIT_LEGACY", None)
        out = []
        import torch; torch.cuda.synchronize()
        t = time.time()
        for i in cams:
            ref = V(i); rng = mvs.depth_range(ch, i, ref)
            out.append(compute_depth_gpu(ref, [V(j) for j in nb[i]], rng[0], rng[1], 192, 9))
        torch.cuda.synchronize()
        res[mode] = (time.time() - t, out)
        print("%s: %.2f s/view (GPU kernel only, %d views)" % (mode, res[mode][0] / len(cams), len(cams)), flush=True)
    same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(res["legacy"][1], res["fast"][1]))
    print("identical depth + confidence:", same)


if __name__ == "__main__":
    main(*sys.argv[1:2])
