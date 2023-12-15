#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid

import math
import numpy as np
from .losses import IOUloss, LandmarksLoss, CrossEntropyLoss2d, DiceLoss
from .network_blocks import BaseConv, DWConv, RFB2, PyramidPooling, FFM, ProtoNet


class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes,
            width=1.0,
            keypoints=0,
            segcls=0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False,
            model_export=False,
            repeat=2,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.keypoints = keypoints
        self.segcls = segcls
        self.coef_dim = 32 if segcls > 0 else 0
        self.decode_in_inference = True  # for deploy, set to False
        self.model_export = model_export

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.seg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        if self.keypoints > 0:
            self.kpts_convs = nn.ModuleList()
            self.lmk_preds = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv
        if self.segcls > 0:
            self.proto_net = ProtoNet(in_channel=int(in_channels[0] * width), coef_dim=self.coef_dim, width=width)
            if self.training:
                self.semantic_seg_conv = nn.Conv2d(int(in_channels[0] * width), self.segcls, kernel_size=1)
            self.seg_preds = nn.ModuleList()
            self.mask_ratio = 4 if len(in_channels) == 3 else 2
            in_channels = in_channels if len(in_channels) == 3 else in_channels[1:]

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ) for _ in range(repeat)
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ) for _ in range(repeat)
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            if self.keypoints > 0:
                # self.kps_sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
                #                        1.07, .87, .87, .89, .89]) / 10.0  # Key points of human body
                self.kps_sigmas = torch.tensor([0.5 for _ in range(self.keypoints)])
                self.lmk_preds.append(
                    nn.Conv2d(
                        in_channels=int(256 * width),
                        out_channels=self.n_anchors * self.keypoints * 3,  # 3
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            if self.segcls > 0:
                self.seg_convs.append(
                    nn.Sequential(
                        *[
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                            Conv(
                                in_channels=int(256 * width),
                                out_channels=int(256 * width),
                                ksize=3,
                                stride=1,
                                act=act,
                            ),
                        ]
                    )
                )
                self.seg_preds.append(
                    nn.Sequential(
                        *[
                            nn.Conv2d(
                                in_channels=int(256 * width),
                                out_channels=self.n_anchors * self.coef_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                            ),
                            nn.Tanh()
                        ]
                    )
                )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.lmk_loss = LandmarksLoss()
        self.cross_entropy_loss2d = CrossEntropyLoss2d(ignore_index=-1)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.expanded_strides = [None] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        if self.segcls > 0:
            for conv in self.seg_preds:
                if isinstance(conv, nn.Conv2d):
                    b = conv.bias.view(self.n_anchors, -1)
                    b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                    conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            nn.init.xavier_uniform_(self.semantic_seg_conv.weight.data)
            self.semantic_seg_conv.bias.data.zero_()

    def forward(self, xin, labels=None, imgs=None, seg_labels=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        seg_proto, semantic_pred = None, None
        if self.segcls > 0:
            seg_proto = self.proto_net(xin[0]).permute(0, 2, 3, 1).contiguous()  # feature map P3   放大2
            if self.training:
                semantic_pred = self.semantic_seg_conv(xin[0])
            xin = xin if len(xin) == 3 else xin[1:]
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_x = x
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            if self.keypoints > 0:
                lmk_output = self.lmk_preds[k](reg_feat)
            if self.segcls > 0:
                seg_x = x
                seg_feat = self.seg_convs[k](seg_x)
                seg_output = self.seg_preds[k](seg_feat)

            if self.training:
                if self.keypoints > 0:
                    output = torch.cat([reg_output, obj_output, cls_output, lmk_output], 1)
                elif self.segcls > 0:
                    output = torch.cat([reg_output, obj_output, cls_output, seg_output], 1)
                else:
                    output = torch.cat([reg_output, obj_output, cls_output], 1)

                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                        .fill_(stride_this_level)
                        .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                if self.keypoints > 0:
                    output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid(), lmk_output], 1)
                elif self.segcls > 0:
                    output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid(), seg_output], 1)
                else:
                    output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
                seg_proto=seg_proto,
                semantic_pred=semantic_pred,
                seg_labels=seg_labels
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:  # semantic_pred  seg_proto
                return self.decode_outputs(outputs, seg_proto=seg_proto, dtype=xin[0].type())
            else:
                return outputs  # , seg_proto

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes + self.keypoints * 3 + self.coef_dim  # 5:x,y,w,h,bs
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride  # cxcy
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride  # wh
        if self.keypoints > 0:
            kpt_conf_grids = torch.zeros_like(grid)[..., 0:1]
            kpt_grids = torch.cat((grid, kpt_conf_grids), dim=2).repeat(1, 1, self.keypoints)
            output[..., -3 * self.keypoints:] = (output[..., -3 * self.keypoints:] + kpt_grids) * stride
        return output, grid

    def decode_outputs(self, outputs, seg_proto, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides  # cxcy
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides  # wh
        if self.keypoints > 0:
            kpt_conf_grids = torch.zeros_like(grids)[..., 0:1]
            kpt_grids = torch.cat((grids, kpt_conf_grids), dim=2).repeat(1, 1, self.keypoints)
            outputs[..., -3 * self.keypoints:] = (outputs[..., -3 * self.keypoints:] + kpt_grids) * strides
        return outputs, seg_proto

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
            seg_labels=None,
            seg_proto=None,
            semantic_pred=None
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:5 + self.num_classes]  # [batch, n_anchors_all, n_cls]
        if self.keypoints > 0:
            lmk_preds = outputs[:, :, 5 + self.num_classes:]  # [batch, n_anchors_all, 10]
        if self.segcls > 0:
            seg_preds = outputs[:, :, 5 + self.num_classes:]  # [batch, n_anchors_all, self.coef_dim]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        cls_targets_ls = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        lmk_targets = []
        seg_targets = []
        fg_masks = []
        num_fg = 0.0
        num_gts = 0.0
        batch_size, _, self.im_h, self.im_w = imgs.shape
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                if self.keypoints > 0:
                    lmk_target = outputs.new_zeros((0, self.keypoints * 2))
                if self.segcls > 0:
                    seg_target = seg_labels[batch_idx]
                    cls_targets_ls.append(cls_target)

            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                if self.keypoints > 0:
                    gt_lmks = labels[batch_idx, :num_gt, -2 * self.keypoints:]
                if self.segcls > 0:
                    gt_segs = seg_labels[batch_idx]
                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img
                if self.segcls > 0:
                    cls_targets_ls.append(gt_matched_classes)
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.keypoints > 0:
                    lmk_target = gt_lmks[matched_gt_inds]
                if self.segcls > 0:
                    seg_target = gt_segs
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            if self.keypoints > 0:
                lmk_targets.append(lmk_target)
            if self.segcls > 0:
                seg_targets.append(seg_target.unsqueeze(0))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        if self.keypoints > 0:
            lmk_targets = torch.cat(lmk_targets, 0)
        if self.segcls > 0:
            seg_targets = torch.cat(seg_targets, 0)
            reg_targets_ls = reg_targets.copy()
            fg_masks_ls = fg_masks.copy()
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        if self.keypoints > 0:
            loss_kpts, loss_kpts_vis = self.kpts_loss(
                lmk_preds.view(-1, self.keypoints * 3)[fg_masks], lmk_targets, reg_targets)
            loss_lmk = 5.0 * loss_kpts.sum() / num_fg + loss_kpts_vis.sum() / num_fg
        else:
            loss_lmk = 0

        if self.segcls > 0:
            loss_s, loss_m, total_pos_num = 0, 0, 0
            batch_size, num_classes, mask_h, mask_w = semantic_pred.size()
            seg_targets = seg_targets.squeeze(1)
            b, h, w = seg_targets.size()
            seg_targets_ls = []
            for b_ in range(b):
                seg_targets2 = torch.zeros(self.segcls, h, w)
                for cls in range(1, self.segcls):
                    seg_targets2[cls][seg_targets[b_] == cls] = 1
                seg_targets_ls.append(seg_targets2)
            seg_labels_section = torch.stack(seg_targets_ls, dim=0).cuda()
            semantic_section = F.interpolate(seg_labels_section, (mask_h, mask_w), mode='bilinear',
                                             align_corners=False).gt(0.5)

            semantic_target = torch.zeros((batch_size, mask_h, mask_w), requires_grad=False,
                                          device=semantic_pred.device)
            for b_ in range(b):
                for cls in range(1, self.segcls):
                    semantic_target[b_][semantic_section[b_, cls]] = cls
            loss_s = self.cross_entropy_loss2d(semantic_pred, semantic_target.long())

            proto_h, proto_w = seg_proto.shape[1:3]  # 320--> 80,80,32
            for bb in range(batch_size):
                fg_mask, pos_anchor_box = fg_masks_ls[bb], reg_targets_ls[bb]
                if pos_anchor_box.size(0) == 0:
                    continue
                cur_class_gt = cls_targets_ls[bb].long()
                proto_section = F.interpolate(seg_labels_section[bb].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).gt(0.5).float().squeeze(0)

                pos_coef = seg_preds[bb].view(-1, self.coef_dim)[fg_mask]
                mask_p = seg_proto[bb] @ pos_coef.t()

                mask_p, anchor_area = self.crop(mask_p, pos_anchor_box.clone())

                proto_section_ls = []
                for cls in cur_class_gt:
                    proto_section_ls.append(proto_section[cls + 1])
                proto_section = torch.stack(proto_section_ls, dim=-1).contiguous()
                proto_section, _ = self.crop(proto_section, pos_anchor_box.clone())

                mask_loss = self.bcewithlog_loss(mask_p, proto_section)
                mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area
                loss_m += torch.sum(mask_loss)
            loss_m = loss_m / proto_h / proto_w / num_fg * 6.125

            loss_seg = loss_m + loss_s
        else:
            loss_seg = 0

        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        lmk_weight = 1.0
        seg_weight = 1.0

        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + \
               loss_lmk * lmk_weight + \
               seg_weight * loss_seg

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            loss_lmk,
            loss_seg,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def crop(self, masks, boxes, padding=1):
        box_corner = boxes.new(boxes.shape)
        w, h = boxes[..., 2], boxes[..., 3]
        area = w / self.mask_ratio * h / self.mask_ratio
        box_corner[..., 0] = (boxes[..., 0] - w / 2)
        box_corner[..., 1] = (boxes[..., 1] - h / 2)
        box_corner[..., 2] = (boxes[..., 0] + w / 2)
        box_corner[..., 3] = (boxes[..., 1] + h / 2)

        h, w, n = masks.size()  # 80,80,-1
        box_corner[..., [0, 2]] /= w * self.mask_ratio
        box_corner[..., [1, 3]] /= h * self.mask_ratio
        x1, x2 = self.sanitize_coordinates(box_corner[:, 0], box_corner[:, 2], w, padding)
        y1, y2 = self.sanitize_coordinates(box_corner[:, 1], box_corner[:, 3], h, padding)

        rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
        cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

        masks_left = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up = cols >= y1.view(1, 1, -1)
        masks_down = cols < y2.view(1, 1, -1)
        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask.float(), area

    def sanitize_coordinates(self, _x1, _x2, img_size, padding=0):
        """
        Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
        Also converts from relative to absolute coordinates and casts the results to long tensors.

        Warning: this does things in-place behind the scenes so copy if necessary.
        """
        _x1 = _x1 * img_size
        _x2 = _x2 * img_size

        x1 = torch.min(_x1, _x2)
        x2 = torch.max(_x1, _x2)
        x1 = torch.clamp(x1 - padding, min=0)
        x2 = torch.clamp(x2 + padding, max=img_size)

        return x1, x2

    def kpts_loss(self, kpts_preds, kpts_targets, bbox_targets):
        sigmas = self.kps_sigmas.to(kpts_preds.device)
        kpts_preds_x, kpts_targets_x = kpts_preds[:, 0::3], kpts_targets[:, 0::2]
        kpts_preds_y, kpts_targets_y = kpts_preds[:, 1::3], kpts_targets[:, 1::2]
        kpts_preds_score = kpts_preds[:, 2::3]
        # mask
        kpt_mask = (kpts_targets[:, 0::2] > 0)

        lkptv = self.bcewithlog_loss(kpts_preds_score, kpt_mask.float()).mean(axis=1)
        # OKS based loss
        d = (kpts_preds_x - kpts_targets_x) ** 2 + (kpts_preds_y - kpts_targets_y) ** 2
        bbox_scale = torch.prod(bbox_targets[:, -2:], dim=1, keepdim=True)  # scale derived from bbox gt
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / torch.sum(kpt_mask != 0)
        oks = torch.exp(-d / (bbox_scale * (4 * sigmas) + 1e-9))
        lkpt = kpt_loss_factor * ((1 - oks ** 2) * kpt_mask).mean(axis=1)

        # oks = torch.exp(-d / (2 * (bbox_scale * sigmas / 10.0) ** 2 + 1e-9))
        # lkpt = kpt_loss_factor * ((1 - oks) * kpt_mask).mean()

        return lkpt, lkptv
