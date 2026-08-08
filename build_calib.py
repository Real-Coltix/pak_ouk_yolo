#!/usr/bin/env python3
"""Build an INT8 calibration set for RKNN quantization.

Two things most guides get wrong, both of which cost real accuracy:

1. The calibration images must go through the SAME preprocessing as inference.
   RKNN's dataset loader does a plain resize if the image isn't already
   640x640, which stretches it -- but at runtime we letterbox (aspect-preserved
   + grey 114 padding). Calibrating on stretched images and running on
   letterboxed ones means the quantization ranges were measured on a different
   distribution than production sees.

2. Channel order must match. The model was trained on RGB. We save .npy arrays
   so the bytes handed to the quantizer are unambiguous, instead of relying on
   cv2.imread's BGR convention inside the toolkit.

3. Paths in dataset.txt are written RELATIVE, not absolute. rknn-toolkit2
   resolves each entry against the DIRECTORY CONTAINING dataset.txt (not the
   cwd of the calling process), so a relative list next to calib/ works both
   natively and inside a container that mounts the repo at a different prefix
   (e.g. /work). Absolute paths bake in the mount point and fail with
   FileNotFoundError: /work/calib/calib_0000.npy the moment the same
   dataset.txt is reused outside that container. Corollary: a subset list must
   live in the same directory as dataset.txt, or its relative entries break.

Note: when dataset.txt lists .npy files, rknn-toolkit2 bypasses its image
loader and requires the array in the ONNX input layout -- NCHW (1,3,H,W).
Feeding (H,W,3) or (1,H,W,3) fails with
    "The input(...) shape (1, 640, 640, 3) is wrong, expect 'nchw' ..."
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

INPUT_SIZE = 640   # default only; override with --imgsz
PAD = 114          # default only; override with --pad


def letterbox_rgb(path, size=INPUT_SIZE, pad=PAD):
    """Identical geometry to YoloDetector._letterbox(), returned as RGB uint8."""
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib")
    ap.add_argument("--list", default="dataset.txt")
    # Must match the --imgsz used by export_rknn_onnx.py. A mismatch here is
    # silent: the quantizer happily calibrates 640x640 ranges for a 960x960
    # model and you only see it as unexplained accuracy loss.
    ap.add_argument("--imgsz", type=int, default=INPUT_SIZE)
    ap.add_argument("--pad", type=int, default=PAD)
    # Rockchip recommends 20-200 images; more lengthens quantization without
    # necessarily improving accuracy (RKNN SDK User Guide V2.3.2, 6.2.3).
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--globs", nargs="+", default=[
        "datasets/diverse_airborne_6/**/*.jpg",
        "datasets/air_defense_yolov8/**/*.jpg",
    ])
    args = ap.parse_args()

    pools = []
    for g in args.globs:
        found = sorted(glob.glob(g, recursive=True))
        pools.append(found)
        print(f"{len(found):6d} candidates  {g}")

    rng = random.Random(args.seed)
    picks, per = [], max(1, args.n // max(1, len([p for p in pools if p])))
    for pool in pools:
        if pool:
            picks += rng.sample(pool, min(per, len(pool)))
    rng.shuffle(picks)
    picks = picks[: args.n]

    os.makedirs(args.out, exist_ok=True)
    written = []
    for i, p in enumerate(picks):
        arr = letterbox_rgb(p, args.imgsz, args.pad)
        if arr is None:
            continue
        dst = os.path.join(args.out, f"calib_{i:04d}.npy")
        # HWC -> NCHW, the layout the quantizer expects for .npy inputs
        np.save(dst, arr.transpose(2, 0, 1)[None])
        written.append(dst)

    with open(args.list, "w") as f:
        f.write("\n".join(written) + "\n")

    print(f"\nwrote {len(written)} calibration tensors -> {args.out}/")
    print(f"index: {args.list}")
    a = np.load(written[0])
    print(f"sample: shape={a.shape} dtype={a.dtype} min={a.min()} max={a.max()}")


if __name__ == "__main__":
    main()
