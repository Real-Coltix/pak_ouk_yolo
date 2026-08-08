#!/usr/bin/env python3
"""Export best_5s.pt to an RKNN-friendly ONNX (9 outputs, no DFL/decode in-graph).

Why not the airockchip/ultralytics_yolov8 fork: that fork pins ultralytics 8.0.x,
whose Detect head has a different module layout than the 8.4.x head this
checkpoint was trained with (8.4.56) -- loading the .pt there fails on the
state_dict. This script reproduces the fork's RKNN export branch verbatim
against the locally installed ultralytics instead.

Reference (airockchip/ultralytics_yolov8, ultralytics/nn/modules/head.py):

    if self.export and self.format == 'rknn':
        y = []
        for i in range(self.nl):
            y.append(self.cv2[i](x[i]))
            cls = torch.sigmoid(self.cv3[i](x[i]))
            cls_sum = torch.clamp(cls.sum(1, keepdim=True), 0, 1)
            y.append(cls)
            y.append(cls_sum)
        return y
"""
import argparse
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect


def rknn_forward(self, x):
    """RKNN export head: 3 branches x (box_raw, cls_sigmoid, cls_sum)."""
    y = []
    for i in range(self.nl):
        y.append(self.cv2[i](x[i]))
        cls = torch.sigmoid(self.cv3[i](x[i]))
        cls_sum = torch.clamp(cls.sum(1, keepdim=True), 0, 1)
        y.append(cls)
        y.append(cls_sum)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/root/best_5s.pt")
    ap.add_argument("--out", default="best_5s_rknn.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    model = YOLO(args.weights).model
    model.eval()
    model.float()
    model.fuse()

    head = model.model[-1]
    assert isinstance(head, Detect), f"unexpected head {type(head).__name__}"
    nl, nc, reg_max = head.nl, head.nc, head.reg_max
    strides = [int(s) for s in head.stride.tolist()]

    # switch every head into export mode, then install the RKNN branch
    for m in model.modules():
        if isinstance(m, Detect):
            m.export = True
            m.format = "rknn"
    Detect.forward = rknn_forward

    names = []
    for i, s in enumerate(strides):
        names += [f"box{i}_s{s}", f"cls{i}_s{s}", f"sum{i}_s{s}"]

    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        outs = model(dummy)
    print(f"head: nl={nl} nc={nc} reg_max={reg_max} strides={strides}")
    for n, o in zip(names, outs):
        print(f"  {n:14s} {tuple(o.shape)}")

    torch.onnx.export(
        model,
        dummy,
        args.out,
        opset_version=args.opset,
        input_names=["images"],
        output_names=names,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
