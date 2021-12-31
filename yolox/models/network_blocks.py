#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch, math
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self,
            in_channels,
            out_channels,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
            self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
            self,
            in_channels,
            out_channels,
            n=1,
            shortcut=True,
            expansion=0.5,
            depthwise=False,
            act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


############# instance ###########
class RFB2(nn.Module):
    def __init__(self, in_planes, out_planes, map_reduce=4, d=[2, 3], has_globel=False):
        super(RFB2, self).__init__()
        self.out_channels = out_planes
        self.has_globel = has_globel
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BaseConv(in_channels=in_planes, out_channels=inter_planes, ksize=1, stride=1),
            BaseConv(in_channels=inter_planes, out_channels=inter_planes, ksize=3, stride=1)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[0], dilation=d[0], bias=False),
            nn.BatchNorm2d(inter_planes),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1, padding=d[1], dilation=d[1], bias=False),
            nn.BatchNorm2d(inter_planes),
            nn.SiLU()
        )
        self.branch3 = nn.Sequential(
            BaseConv(in_planes, inter_planes, ksize=1, stride=1),
        )
        if self.has_globel:
            self.branch4 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                BaseConv(inter_planes, inter_planes, ksize=1),
            )
        self.ConvLinear = BaseConv(int(5 * inter_planes) if has_globel else int(4 * inter_planes), out_planes, ksize=1,
                                   stride=1)

    def forward(self, x):
        x3 = self.branch3(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        if not self.has_globel:
            out = self.ConvLinear(torch.cat([x0, x1, x2, x3], 1))
        else:
            x4 = F.interpolate(self.branch4(x2), (x.shape[2], x.shape[3]), mode='nearest')
            out = self.ConvLinear(torch.cat([x0, x1, x2, x3, x4], 1))
        return out


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, in_channels, k=[1, 2, 3, 6]):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(k[0])
        self.pool2 = nn.AdaptiveAvgPool2d(k[1])
        self.pool3 = nn.AdaptiveAvgPool2d(k[2])
        self.pool4 = nn.AdaptiveAvgPool2d(k[3])
        print('*** AdaptiveAvgPool2d for train ***' * 5)
        # self.pool1 = nn.AvgPool2d(kernel_size=80, stride=80)
        # self.pool2 = nn.AvgPool2d(kernel_size=40, stride=40)
        # self.pool3 = nn.AvgPool2d(kernel_size=28, stride=26)
        # self.pool4 = nn.AvgPool2d(kernel_size=15, stride=13)
        # print('**** for onnx ****'*500)

        out_channels = in_channels // 4
        self.conv1 = BaseConv(in_channels, out_channels, ksize=1, stride=1)
        self.conv2 = BaseConv(in_channels, out_channels, ksize=1, stride=1)
        self.conv3 = BaseConv(in_channels, out_channels, ksize=1, stride=1)
        self.conv4 = BaseConv(in_channels, out_channels, ksize=1, stride=1)

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode='bilinear', align_corners=True)

        return torch.cat((x, feat1, feat2, feat3, feat4,), 1)


class FFM(nn.Module):
    def __init__(self, in_chan, out_chan, reduction=1, is_cat=True, k=1):
        super(FFM, self).__init__()
        self.convblk = BaseConv(in_chan, out_chan, ksize=k, stride=1)
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                               nn.Conv2d(out_chan, out_chan // reduction,
                                                         kernel_size=1, stride=1, padding=0, bias=False),
                                               nn.SiLU(inplace=True),
                                               nn.Conv2d(out_chan // reduction, out_chan,
                                                         kernel_size=1, stride=1, padding=0, bias=False),
                                               nn.Sigmoid(),
                                               )
        self.is_cat = is_cat

    def forward(self, fspfcp):
        fcat = torch.cat(fspfcp, dim=1) if self.is_cat else fspfcp
        feat = self.convblk(fcat)
        atten = self.channel_attention(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


########### instance2 ##########
import collections
from itertools import repeat
from torch.autograd import Function


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class FeatureAlign(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAlign, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(4,
                                     deformable_groups * offset_channels,
                                     1,
                                     bias=False)
        self.conv_adaption = BaseConv(in_channels,
                                      out_channels,
                                      ksize=kernel_size,
                                      stride=1,
                                      groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.GroupNorm(32, in_channels)

    def init_weights(self, bias_value=0):
        torch.nn.init.normal_(self.conv_offset.weight, std=0.0)
        torch.nn.init.normal_(self.conv_adaption.weight, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.norm(self.conv_adaption(x, offset)))
        return x


class ProtoNet(nn.Module):
    def __init__(self, in_channel=256, coef_dim=32, width=1):
        super().__init__()
        self.proto1 = nn.Sequential(nn.Conv2d(in_channel, int(256 * width), kernel_size=3, stride=1, padding=1),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(int(256 * width), int(256 * width), kernel_size=3, stride=1, padding=1),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(int(256 * width), int(256 * width), kernel_size=3, stride=1, padding=1),
                                    nn.SiLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proto2 = nn.Sequential(nn.Conv2d(int(256 * width), int(256 * width), kernel_size=3, stride=1, padding=1),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(int(256 * width), coef_dim, kernel_size=1, stride=1),
                                    nn.SiLU(inplace=True))

    def forward(self, x):
        x = self.proto1(x)
        x = self.upsample(x)
        x = self.proto2(x)
        return x
