"""Download every network weight shapekit can use into one folder (TORCH_HOME layout), so the portable build
works offline:  python -m shapekit.tools.prefetch_models models
"""
import os
import sys


def main(dest="models"):
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)
    os.environ["TORCH_HOME"] = dest
    import torch
    torch.hub.set_dir(os.path.join(dest, "hub"))
    print("TORCH_HOME =", dest, flush=True)
    from lightglue import SuperPoint, ALIKED, DISK, LightGlue
    for name, ctor in (("ALIKED", lambda: ALIKED(model_name="aliked-n16")), ("SuperPoint", SuperPoint), ("DISK", DISK)):
        ctor(); print("  ok", name, flush=True)
    for feat in ("aliked", "superpoint", "disk"):
        LightGlue(features=feat); print("  ok LightGlue for", feat, flush=True)
    from ..learned.retrieval import GlobalDescriptor
    GlobalDescriptor("salad"); print("  ok DINOv2 + SALAD", flush=True)
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1); print("  ok DeepLabV3 (AI masks)", flush=True)
    total = 0
    for root, _, files in os.walk(dest):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    print("models folder: %.0f MB" % (total / 1e6), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
