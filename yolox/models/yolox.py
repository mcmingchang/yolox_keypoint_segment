#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolo_pafpn_slim import YOLOPAFPNSLIM
from .yolov7_tiny import YOLO7TINY
from yolox.utils.fuse_model import fuse_conv_and_bn
from yolox.models.network_blocks import BaseConv

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        # for yolov7
        if isinstance(self.backbone, YOLO7TINY):
            print('Fusing layers... ')
            for m in self.backbone.modules():
                if type(m) is BaseConv and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.fuseforward  # update forward

    def forward(self, x, targets=None, seg_targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, lmk_loss, seg_loss, num_fg = self.head(
                fpn_outs, targets, x, seg_targets
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            seg_output = None
            if self.head.keypoints > 0:
                outputs['kp_loss'] = lmk_loss
            if self.head.segcls > 0:
                outputs['seg_loss'] = seg_loss
        else:
            if self.head.decode_in_inference is False:
                outputs = self.head(fpn_outs)
                return outputs
            outputs, seg_output = self.head(fpn_outs)
        return outputs, seg_output
