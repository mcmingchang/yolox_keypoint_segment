#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class LandmarksLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha

    def forward(self, pred, truel):
        mask = truel > 0
        loss = self.loss_fcn(pred*mask, truel*mask)
        return loss / (torch.sum(mask) + 10e-14)


class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1, aux_num=2,
                 aux=False, aux_weight=0.1, weight=None,
                 ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)
        self.aux_num = aux_num

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)  # CrossEntropyLoss
        elif not self.se_loss:
            if self.aux_num == 2:
                pred1, pred2, pred3, target = tuple(inputs)
                loss1 = super(SegmentationLosses, self).forward(pred1, target)
                loss2 = super(SegmentationLosses, self).forward(pred2, target)
                loss3 = super(SegmentationLosses, self).forward(pred3, target)
                return loss1 + self.aux_weight*1.5 * loss2 + self.aux_weight/2.0 * loss3
            else:
                assert self.aux_num == 1
                pred1, pred2, target = tuple(inputs)
                loss1 = super(SegmentationLosses, self).forward(pred1, target)
                loss2 = super(SegmentationLosses, self).forward(pred2, target)
                return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3


class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, ignore_index=0,
                 reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction,
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets, do_rmi=None):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def _gather_feat(feat, ind):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


# def Dice_loss(inputs, target, beta=1, smooth=1e-5):
#     n, c, h, w = inputs.size()
#     nt, ht, wt, ct = target.size()
#     if h != ht and w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
#
#     temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
#     temp_target = target.view(n, -1, ct)
#
#     # --------------------------------------------#
#     #   计算dice loss
#     # --------------------------------------------#
#     tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
#     fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
#     fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
#
#     score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
#     dice_loss = 1 - torch.mean(score)
#     return dice_loss



class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        iflat = input.permute(0, 2, 3, 1).contiguous().view(-1)
        tflat = target.permute(0, 2, 3, 1).contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + self.smooth) / ((iflat * iflat).sum() + (tflat * tflat).sum() + self.smooth))