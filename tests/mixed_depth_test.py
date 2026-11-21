"""Regression test for depth maps with neighbours of different image sizes (portrait + landscape videos)."""
import os, sys, time
import numpy as np


def main(same_proj="projects/gpu_test", mixed_proj="projects/area_d"):
    from shapekit.project import Project
    from shapekit import mvs
    from shapekit.learned.mvs_gpu import compute_depth_gpu
    # 1) same-size neighbours: new code must equal the previous version of the function bit-for-bit
    from tests._mvs_gpu_prev import compute_depth_gpu as compute_depth_gpu_prev
    ch = Project.open(same_proj).chunk()
    nb = mvs.select_neighbors(ch, 5)
    ok = True
    for i in [i for i in ch.aligned_cameras() if nb.get(i)][:4]:
        ref = mvs.View(ch, i, 0.5); rng = mvs.depth_range(ch, i, ref)
        nbrs = [mvs.View(ch, j, 0.5) for j in nb[i]]
        d, conf = compute_depth_gpu(ref, nbrs, rng[0], rng[1], 192, 9)
        d0, c0 = compute_depth_gpu_prev(ref, nbrs, rng[0], rng[1], 192, 9)
        same = np.array_equal(d, d0) and np.array_equal(conf, c0)
        ok &= same
        print("same-size view %d: identical to previous version: %s" % (i, same), flush=True)
    # 2) mixed sizes
    ch = Project.open(mixed_proj).chunk()
    nb = mvs.select_neighbors(ch, 5)
    sizes = {i: (ch.sensors[ch.cameras[i].sensor].width, ch.sensors[ch.cameras[i].sensor].height) for i in ch.aligned_cameras()}
    mixed = [i for i in ch.aligned_cameras() if nb.get(i) and len({sizes[j] for j in nb[i]} | {sizes[i]}) > 1]
    print("\n%s: %d of %d views have neighbours of a different size" % (mixed_proj, len(mixed), len(sizes)), flush=True)
    for i in mixed[:3]:
        ref = mvs.View(ch, i, 0.5); rng = mvs.depth_range(ch, i, ref)
        nbrs = [mvs.View(ch, j, 0.5) for j in nb[i]]
        t = time.time(); dg, cg = compute_depth_gpu(ref, nbrs, rng[0], rng[1], 192, 9); tg = time.time() - t
        t = time.time(); dc, cc = mvs.compute_depth(ref, nbrs, rng[0], rng[1], 192, 9); tc = time.time() - t
        both = (dg > 0) & (dc > 0)
        rel = np.median(np.abs(dg[both] - dc[both]) / dc[both]) if both.any() else float("nan")
        agree = (np.abs(dg[both] - dc[both]) / dc[both] < 0.01).mean() if both.any() else 0
        print("mixed view %d (%dx%d, nbr sizes %s): GPU %.0f%% valid in %.1fs | CPU %.0f%% valid in %.1fs | "
              "%.1f%% of pixels agree within 1%% depth (median rel diff %.1e)"
              % (i, ref.w, ref.h, sorted({(n.w, n.h) for n in nbrs}), 100 * (dg > 0).mean(), tg,
                 100 * (dc > 0).mean(), tc, 100 * agree, rel), flush=True)
        ok &= (dg > 0).mean() > 0.3 and agree > 0.95
    print("\nMIXED DEPTH TEST %s" % ("OK" if ok else "FAILED"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
