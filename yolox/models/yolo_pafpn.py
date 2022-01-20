#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .coatnet import coatnet_0, coatnet_2
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        img_channel=3,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        backbone_name='CSPDarknet',
        input_size=(320, 320)
    ):
        super().__init__()
        if backbone_name == 'CoAtNet':
            self.backbone = coatnet_2(img_shape=input_size, img_channel=img_channel, dep_mul=depth,
                                      wid_mul=width, out_features=in_features)
        else:
            self.backbone = CSPDarknet(img_channel, depth, width, depthwise=depthwise,
                                       act=act, out_features=in_features)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[-1] * width), int(in_channels[-2] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[-2] * width),
            int(in_channels[-2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[-2] * width), int(in_channels[-3] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[-3] * width),
            int(in_channels[-3] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[-3] * width), int(in_channels[-3] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[-3] * width),
            int(in_channels[-2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[-2] * width), int(in_channels[-2] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[-2] * width),
            int(in_channels[-1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        if len(self.in_channels) == 4:
            self.reduce_conv2 = BaseConv(
                        int(in_channels[-3] * width), int(in_channels[-4] * width), 1, 1, act=act
                    )

            self.C3_p2 = CSPLayer(
                        int(2 * in_channels[-4] * width),
                        int(in_channels[-4] * width),
                        round(3 * depth),
                        False,
                        depthwise=depthwise,
                        act=act,
                    )
            self.bu_conv3 = Conv(
                int(in_channels[-4] * width), int(in_channels[-4] * width), 3, 2, act=act
            )
            self.C3_n2 = CSPLayer(
                int(2 * in_channels[-4] * width),
                int(in_channels[-3] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)

        features = [out_features[f] for f in self.in_features]
        if len(features) == 3:
            [x2, x1, x0] = features  # 尺寸从大到小
            fpn_out0 = self.lateral_conv0(x0)  # in:512,10,10  out:v,10,10
            f_out0 = self.upsample(fpn_out0)  # in:256,10,10  out:256,20,20
            f_out0 = torch.cat([f_out0, x1], 1)  # in:256,20,20  out:512,20,20
            f_out0 = self.C3_p4(f_out0)  # in:512,20,20  out:256,20,20

            fpn_out1 = self.reduce_conv1(f_out0)  # in:256,20,20  out:128,20,20
            f_out1 = self.upsample(fpn_out1)  # in:128,20,20  out:128,40,40
            f_out1 = torch.cat([f_out1, x2], 1)  # in::128,40,40  out:256,40,40
            pan_out2 = self.C3_p3(f_out1)  # in:256,40,40  out:128,40,40

            p_out1 = self.bu_conv2(pan_out2)  # in:128,40,40  out:128,20,20
            p_out1 = torch.cat([p_out1, fpn_out1], 1)  # int:128,20,20  out:256,20,20
            pan_out1 = self.C3_n3(p_out1)  # in:256,20,20  out:256,20,20

            p_out0 = self.bu_conv1(pan_out1)  # in:256,20,20  out:256,10,10
            p_out0 = torch.cat([p_out0, fpn_out0], 1)  # in:256,10,10  out:512,10,10
            pan_out0 = self.C3_n4(p_out0)  # in:512,10,10  out:512,10,10

            outputs = (pan_out2, pan_out1, pan_out0)
        else:
            [x3, x2, x1, x0] = features  # 尺寸从大到小
            fpn_out0 = self.lateral_conv0(x0)  # in:512,10,10  out:v,10,10
            f_out0 = self.upsample(fpn_out0)  # in:256,10,10  out:256,20,20
            f_out0 = torch.cat([f_out0, x1], 1)  # in:256,20,20  out:512,20,20
            f_out0 = self.C3_p4(f_out0)  # in:512,20,20  out:256,20,20

            fpn_out1 = self.reduce_conv1(f_out0)  # in:256,20,20  out:128,20,20
            f_out1 = self.upsample(fpn_out1)  # in:128,20,20  out:128,40,40
            f_out1 = torch.cat([f_out1, x2], 1)  # in::128,40,40  out:256,40,40
            f_out1 = self.C3_p3(f_out1)  # in:256,40,40  out:128,40,40

            fpn_out2 = self.reduce_conv2(f_out1)  # in:128,40,40  out:64,40,40
            f_out2 = self.upsample(fpn_out2)  # in:64,40,40  out:64,80,80
            f_out2 = torch.cat([f_out2, x3], 1)  # in::64,80,80  out:128,80,80
            pan_out3 = self.C3_p2(f_out2)  # in:128,80,80  out:64,80,80

            p_out2 = self.bu_conv3(pan_out3)  # in:64,80,80  out:64,40,40
            p_out2 = torch.cat([p_out2, fpn_out2], 1)  # int:64,40,40  out:128,40,40
            pan_out2 = self.C3_n2(p_out2)  # in:128,40,40  out:128,40,40

            p_out1 = self.bu_conv2(pan_out2)  # in:128,40,40  out:128,20,20
            p_out1 = torch.cat([p_out1, fpn_out1], 1)  # int:128,20,20  out:256,20,20
            pan_out1 = self.C3_n3(p_out1)  # in:256,20,20  out:256,20,20

            p_out0 = self.bu_conv1(pan_out1)  # in:256,20,20  out:256,10,10
            p_out0 = torch.cat([p_out0, fpn_out0], 1)  # in:256,10,10  out:512,10,10
            pan_out0 = self.C3_n4(p_out0)  # in:512,10,10  out:512,10,10

            outputs = (pan_out3, pan_out2, pan_out1, pan_out0)

        return outputs
