#!/usr/bin/env python3
"""Measure what INT8 quantization cost, by running the .rknn in the toolkit's
simulator and diffing against the FP32 ONNX on the same images.

Generic over nc / imgsz / number of detect layers.

Run on the HOST (x86_64). target=None => simulator, no board required.
The simulator is bit-accurate for the quantized arithmetic but does NOT
measure latency; for real FPS you must run eval_perf() on the board.
"""
import argparse
import glob
import json

import cv2
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


def letterbox_rgb(path, size):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def dfl_np(position):
    n, c, h, w = position.shape
    mc = c // 4
    y = position.reshape(n, 4, mc, h, w).astype(np.float32)
    y = y - y.max(axis=2, keepdims=True)
    y = np.exp(y)
    y /= y.sum(axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def box_process(position, size):
    gh, gw = position.shape[2:4]
    col, row = np.meshgrid(np.arange(gw), np.arange(gh))
    grid = np.concatenate((col.reshape(1, 1, gh, gw), row.reshape(1, 1, gh, gw)), 1).astype(np.float32)
    stride = np.array([size // gh, size // gw]).reshape(1, 2, 1, 1)
    d = dfl_np(position)
    xy1 = (grid + 0.5 - d[:, 0:2]) * stride
    xy2 = (grid + 0.5 + d[:, 2:4]) * stride
    return np.concatenate((xy1, xy2), 1)


def sp_flatten(a):
    return a.transpose(0, 2, 3, 1).reshape(-1, a.shape[1])


def decode(outs, size):
    """outs: 3*nl tensors -> (xyxy [N,4], best score [N], best class [N])."""
    nl = len(outs) // 3
    boxes, confs = [], []
    for i in range(nl):
        boxes.append(sp_flatten(box_process(outs[3 * i], size)))
        confs.append(sp_flatten(outs[3 * i + 1]))
    b = np.concatenate(boxes)
    c = np.concatenate(confs)
    return b, c.max(axis=-1), c.argmax(axis=-1)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="best_5s_rknn.onnx")
    ap.add_argument("--rknn", default="best_5s_rknn_rk3588_i8.rknn")
    ap.add_argument("--images", default="datasets/**/*.jpg")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dataset", default="dataset.txt")
    ap.add_argument("--platform", default="rk3588")
    ap.add_argument("--algorithm", default="normal")
    ap.add_argument("--json", default=None,
                    help="dump per-frame records here for offline re-analysis")
    ap.add_argument("--from-json", default=None,
                    help="re-report an earlier --json dump; skips the (multi-minute) "
                         "quantization build entirely, no rknn toolkit needed")
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json) as f:
            d = json.load(f)
        print(f"re-analyzing {args.from_json}  (algorithm={d.get('algorithm')}, "
              f"conf={d.get('conf')})\n")
        report(d["records"], d.get("conf", args.conf))
        return

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    onames = [o.name for o in sess.get_outputs()]

    # NOTE: a .rknn loaded with load_rknn() cannot run in the simulator --
    # the toolkit refuses with "not support inference on the simulator".
    # To simulate we must rebuild in-session with identical config, which
    # produces the same quantized graph that convert_rknn.py exported.
    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
        quantized_dtype="w8a8",
        quantized_algorithm=args.algorithm,
        quantized_method="channel",
        optimization_level=3,
        quant_img_RGB2BGR=False,
    )
    assert rknn.load_onnx(model=args.onnx) == 0, "load_onnx failed"
    assert rknn.build(do_quantization=True, dataset=args.dataset) == 0, "build failed"
    assert rknn.init_runtime(target=None) == 0, "init_runtime (simulator) failed"

    files = sorted(glob.glob(args.images, recursive=True))
    step = max(1, len(files) // args.n)
    files = files[::step][: args.n]

    recs = []
    for p in files:
        rgb = letterbox_rgb(p, args.imgsz)
        if rgb is None:
            continue
        x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        o_f32 = sess.run(onames, {"images": x})
        o_i8 = rknn.inference(inputs=[rgb[None]], data_format="nhwc")

        b_f, s_f, _ = decode(o_f32, args.imgsz)
        b_q, s_q, _ = decode([np.asarray(o) for o in o_i8], args.imgsz)

        hf, hq = s_f >= args.conf, s_q >= args.conf
        r = {
            "file": p,
            "fp32_top": float(s_f.max()),
            "int8_top": float(s_q.max()),
            "fp32_hit": bool(hf.any()),
            "int8_hit": bool(hq.any()),
            # error over ALL anchors is dominated by background noise; the
            # decision-relevant error is over anchors fp32 calls confident.
            "d_all": float(np.abs(s_f - s_q).max()),
            "d_conf": float(np.abs(s_f - s_q)[hf].max()) if hf.any() else None,
            "iou_top": (float(iou(b_f[s_f.argmax()], b_q[s_q.argmax()]))
                        if hf.any() and hq.any() else None),
            # cosine is ill-conditioned when both score maps are ~0 (no target
            # in frame): dot/(|a||b|+eps) -> 0, which looks like total failure
            # but only means "both models agree there is nothing here". Only
            # meaningful on frames where fp32 actually fires.
            "cos": (float(np.dot(s_f, s_q) / (np.linalg.norm(s_f) * np.linalg.norm(s_q) + 1e-12))
                    if hf.any() else None),
        }
        recs.append(r)
    rknn.release()

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"conf": args.conf, "algorithm": args.algorithm,
                       "onnx": args.onnx, "records": recs}, f, indent=1)
        print(f"per-frame records -> {args.json}\n")
    report(recs, args.conf)


def report(recs, conf):
    """Summarize per-frame records. Separated from inference so the JSON dump
    can be re-analyzed without paying for another quantization build."""
    n = len(recs)
    hit_f = [r for r in recs if r["fp32_hit"]]
    both = [r for r in recs if r["fp32_hit"] and r["int8_hit"]]
    missed = [r for r in recs if r["fp32_hit"] and not r["int8_hit"]]
    extra = [r for r in recs if r["int8_hit"] and not r["fp32_hit"]]
    ious = [r["iou_top"] for r in both if r["iou_top"] is not None]
    d_conf = [r["d_conf"] for r in hit_f if r["d_conf"] is not None]
    cos = [r["cos"] for r in hit_f if r["cos"] is not None]
    d_top = [abs(r["fp32_top"] - r["int8_top"]) for r in recs]

    print(f"images evaluated        : {n}   (conf threshold {conf})")
    print(f"frames w/ detection     : fp32 {len(hit_f)}   int8 {len(hit_f) - len(missed) + len(extra)}"
          f"   both {len(both)}")
    print(f"  int8 MISSED           : {len(missed)}/{len(hit_f)} of fp32 detections"
          f"   <- the number that matters")
    print(f"  int8 EXTRA            : {len(extra)}   (fp32 below threshold, int8 above)")
    if d_top:
        print(f"top-score delta         : mean {np.mean(d_top):.4f}   worst {np.max(d_top):.4f}")
    if d_conf:
        print(f"|delta| on fp32-conf anchors: mean {np.mean(d_conf):.4f}   worst {np.max(d_conf):.4f}")
    if cos:
        print(f"score-map cosine (fp32-hit frames only): mean {np.mean(cos):.5f}"
              f"   worst {np.min(cos):.5f}")
    if ious:
        bad = sum(1 for v in ious if v < 0.5)
        print(f"top-box IoU             : mean {np.mean(ious):.4f}   worst {np.min(ious):.4f}")
        print(f"  frames IoU < 0.5      : {bad}/{len(ious)}   <- real localization regressions")

    worst = sorted(recs, key=lambda r: abs(r["fp32_top"] - r["int8_top"]), reverse=True)[:5]
    print("\nworst frames by top-score shift:")
    print(f"  {'fp32':>6s} {'int8':>6s} {'IoU':>6s}  file")
    for r in worst:
        io = f"{r['iou_top']:.3f}" if r["iou_top"] is not None else "  -  "
        print(f"  {r['fp32_top']:6.3f} {r['int8_top']:6.3f} {io:>6s}  {r['file'].split('/')[-1][:52]}")
    for r in missed:
        print(f"MISS: fp32 {r['fp32_top']:.3f} -> int8 {r['int8_top']:.3f}  {r['file']}")


if __name__ == "__main__":
    main()
