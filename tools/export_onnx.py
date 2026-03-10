#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module
from yolox.utils.checkpoint import load_ckpt


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=18, type=int,
        help="onnx opset version (use 18+ for seg model; lower opsets fail on Resize when PyTorch converts down)",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="export ONNX with dynamic batch dimension",
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="export without loading a checkpoint (random-initialized weights). Output ONNX is for smoke-test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        default=True,
        help="decode in inference or not"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    use_random_weights = args.random_weights
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        if not use_random_weights and not os.path.isfile(ckpt_file):
            use_random_weights = True
            logger.warning(
                "No checkpoint at {} and -c not set; exporting with random-initialized weights (smoke-test only).",
                ckpt_file,
            )
    else:
        ckpt_file = args.ckpt

    if use_random_weights:
        logger.warning(
            "Exporting with RANDOM-INITIALIZED weights. The ONNX is for smoke-test/export validation only, not for inference."
        )
        # Label output so it's clear this is not a trained model
        if "_random_weights" not in args.output_name:
            base, ext = os.path.splitext(args.output_name)
            args.output_name = base + "_random_weights" + (ext or ".onnx")
    else:
        # load checkpoint (load_ckpt filters by shape so detection ckpt can partially load into seg model)
        ckpt = torch.load(ckpt_file, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        load_ckpt(model, ckpt)
        logger.info("loading checkpoint done.")

    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference
    model.head.model_export = True
    if exp.model_name == 'yolov7_tiny':
        model.fuse()

    dummy_input = torch.randn(args.batch_size, exp.img_channel, exp.test_size[0], exp.test_size[1])

    outputs = model(dummy_input)
    torch.onnx.export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
