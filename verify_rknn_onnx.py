#!/usr/bin/env python3
"""Prove the 9-output RKNN ONNX is numerically equivalent to the original
1-output ONNX, by decoding both to (xyxy, score) per anchor and diffing.

Decode of the 9-output model reproduces airockchip/rknn_model_zoo
examples/yolov8/python/yolov8.py box_process()/dfl() exactly.
"""
import argparse
import glob

import numpy as np
import onnxruntime as ort

IMG_SIZE = (640, 640)  # (w, h)


def dfl_np(position):
    """Softmax-expectation over reg_max bins. numpy port of model-zoo dfl()."""
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = y - y.max(axis=2, keepdims=True)          # numerically stable softmax
    y = np.exp(y)
    y = y / y.sum(axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def box_process(position):
    """model-zoo box_process(): grid +/- dfl distances, scaled by stride -> xyxy."""
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1).astype(np.float32)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl_np(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def sp_flatten(_in):
    ch = _in.shape[1]
    return _in.transpose(0, 2, 3, 1).reshape(-1, ch)


def decode_rknn(outs):
    """9 outputs -> (xyxy [N,4], score [N]) in original anchor order."""
    boxes, confs = [], []
    for i in range(3):
        boxes.append(sp_flatten(box_process(outs[3 * i])))
        confs.append(sp_flatten(outs[3 * i + 1]))
    return np.concatenate(boxes), np.concatenate(confs).max(axis=-1)


def decode_original(out0):
    """[1,5,8400] cx,cy,w,h,score -> (xyxy [N,4], score [N])."""
    p = out0[0].T                       # [8400, 5]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return xyxy, p[:, 4]


def load_image(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    h, w = img.shape[:2]
    r = min(640 / h, 640 / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((640, 640, 3), 114, np.uint8)
    top, left = (640 - nh) // 2, (640 - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="best_5s.onnx")
    ap.add_argument("--rknn-onnx", default="best_5s_rknn.onnx")
    ap.add_argument("--images", default="datasets/diverse_airborne_6/**/*.jpg")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s_orig = ort.InferenceSession(args.orig, so, providers=["CPUExecutionProvider"])
    s_rknn = ort.InferenceSession(args.rknn_onnx, so, providers=["CPUExecutionProvider"])
    rknn_names = [o.name for o in s_rknn.get_outputs()]

    rng = np.random.default_rng(0)
    cases = [("random-noise", rng.random((1, 3, 640, 640), dtype=np.float32)),
             ("mid-gray", np.full((1, 3, 640, 640), 0.5, np.float32))]
    for p in sorted(glob.glob(args.images, recursive=True))[: args.n]:
        cases.append((p.split("/")[-1][:34], load_image(p)))

    print(f"{'case':36s} {'maxΔbox(px)':>12s} {'maxΔscore':>11s} {'n>0.25':>7s} {'agree':>6s}")
    print("-" * 78)
    worst_b = worst_s = 0.0
    ok = True
    for name, x in cases:
        o_ref = s_orig.run(None, {"images": x})[0]
        o_rk = s_rknn.run(rknn_names, {"images": x})

        b_ref, s_ref = decode_original(o_ref)
        b_rk, s_rk = decode_rknn(o_rk)

        db = float(np.abs(b_ref - b_rk).max())
        ds = float(np.abs(s_ref - s_rk).max())
        n_hit = int((s_ref >= 0.25).sum())
        agree = int((s_ref >= 0.25).sum() == (s_rk >= 0.25).sum())
        worst_b, worst_s = max(worst_b, db), max(worst_s, ds)
        ok &= db < 1e-2 and ds < 1e-4
        print(f"{name:36s} {db:12.3e} {ds:11.3e} {n_hit:7d} {'yes' if agree else 'NO':>6s}")

    print("-" * 78)
    print(f"worst box delta   : {worst_b:.3e} px")
    print(f"worst score delta : {worst_s:.3e}")
    print("\nRESULT:", "EQUIVALENT" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
