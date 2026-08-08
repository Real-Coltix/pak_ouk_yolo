#!/usr/bin/env python3
"""ONNX (RKNN-friendly, 3*nl outputs) -> .rknn for Rockchip NPU.

Generic: works for any Ultralytics detect model exported with
export_rknn_onnx.py, any nc / imgsz / variant.

Run on an x86_64 Linux HOST (not the board), Python 3.6-3.12.
"""
import argparse
import sys

from rknn.api import RKNN

# RK3588S is the reduced-IO variant of RK3588: same 3-core 6 TOPS NPU,
# same instruction set. rknn-toolkit2 has no separate "rk3588s" target --
# 'rk3588' is correct for both.
PLATFORM_ALIASES = {"rk3588s": "rk3588"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="best_5s_rknn.onnx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--platform", default="rk3588")
    ap.add_argument("--dtype", choices=["i8", "fp"], default="i8")
    ap.add_argument("--dataset", default="dataset.txt")
    ap.add_argument("--algorithm", choices=["normal", "mmse", "kl_divergence"],
                    default="normal")
    ap.add_argument("--hybrid", action="store_true",
                    help="auto_hybrid: let the toolkit keep accuracy-critical "
                         "layers in float when INT8 cosine drops below thresh")
    ap.add_argument("--single-core", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    platform = PLATFORM_ALIASES.get(args.platform.lower(), args.platform.lower())
    quant = args.dtype == "i8"
    out = args.out or args.onnx.replace(".onnx", f"_{platform}_{args.dtype}.rknn")

    rknn = RKNN(verbose=args.verbose)

    # mean/std are applied by the NPU itself, so the board feeds raw uint8
    # pixels and skips the float conversion + /255 on the CPU entirely.
    # [[0,0,0]] / [[255,255,255]] reproduces the usual x/255.0.
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform,
        quantized_dtype="w8a8",
        quantized_algorithm=args.algorithm,
        quantized_method="channel",
        optimization_level=3,
        quant_img_RGB2BGR=False,   # we hand the quantizer .npy in the exact order used at runtime
        single_core_mode=args.single_core,
    )

    if rknn.load_onnx(model=args.onnx) != 0:
        sys.exit("load_onnx failed")

    ret = rknn.build(do_quantization=quant,
                     dataset=args.dataset if quant else None,
                     auto_hybrid=args.hybrid)
    if ret != 0:
        sys.exit("build failed")

    if rknn.export_rknn(out) != 0:
        sys.exit("export_rknn failed")

    print(f"\nOK -> {out}")
    rknn.release()


if __name__ == "__main__":
    main()
