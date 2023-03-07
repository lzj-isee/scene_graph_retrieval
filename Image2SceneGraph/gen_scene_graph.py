# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
# from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from inference import inference


def main():
    parser = argparse.ArgumentParser(description="From image to scene graph")
    parser.add_argument(
        "--config-file",
        default="/home/lzj/sgg/configs/e2e_relation_X_101_32_8_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    # merge the configs from file and command_line
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # build model and load the checkpoints
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    # if cfg.MODEL.MASK_ON:
    #     iou_types = iou_types + ("segm",)
    # if cfg.MODEL.KEYPOINT_ON:
    #     iou_types = iou_types + ("keypoints",)
    # if cfg.MODEL.RELATION_ON:
    #     iou_types = iou_types + ("relations", )
    # if cfg.MODEL.ATTRIBUTE_ON:
    #     iou_types = iou_types + ("attributes", )
    # output_folders = [None] * len(cfg.DATASETS.TEST)
    # dataset_names = cfg.DATASETS.TEST
    # if cfg.OUTPUT_DIR:
    #     for idx, dataset_name in enumerate(dataset_names):
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    #         mkdir(output_folder)
    #         output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)
    for data_loader_val in data_loaders_val:
        inference(
            cfg,
            model,
            data_loader_val,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE
        )
        synchronize()

if __name__ == '__main__':
    main()
