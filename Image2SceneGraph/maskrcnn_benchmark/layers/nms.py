# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C
import torch

# from apex import amp

# Only valid with fp32 inputs - give AMP the hint
# nms = amp.float_function(_C.nms)
def nms(boxes, score, nms_thresh):
    with torch.cuda.amp.autocast(False):
        return _C.nms(boxes, score, nms_thresh)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
