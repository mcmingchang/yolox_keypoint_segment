#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np
from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:计算候选框，包括以下5项
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    # 放大前的框1，放大后的框2，像素，纵横比，面积比
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
    )  # candidates


def random_perspective(
        img,
        targets=(),
        degrees=10,
        translate=0.1,
        scale=0.1,
        shear=10,
        perspective=0.0,
        border=(0, 0),
        keypoints=0,
        segcls=0,
        seg=np.array([])
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2  # border=[-input_h // 2, -input_w // 2],

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
            random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:  # False
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # this
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )
            if segcls > 0:
                seg = cv2.warpAffine(
                    seg, M[:2], dsize=(int(width), int(height)), borderValue=(0, 0, 0), flags=cv2.INTER_NEAREST
                )

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:  # False
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale 除以w
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        ind = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[ind]
        targets[:, :4] = xy[ind]
        m = len(targets)
        if keypoints > 0 and m > 0:
            landmarks = np.ones((m * keypoints, 3))
            landmarks[:, :2] = targets[:, -2 * keypoints:].reshape(m * keypoints, 2)
            mask_landmarks = [np.array(x > 0, dtype=np.int32) for x in landmarks]
            landmarks = landmarks @ M.T  # transform
            landmarks = np.array([x * y + y - 1 for x, y in zip(landmarks, mask_landmarks)])

            if perspective:  # False
                mask_landmarks = np.array([np.array(x != -1, dtype=np.int32) for x in landmarks[:, :2]]).reshape(m,
                                                                                                                 2 * keypoints)
                landmarks = (landmarks[:, :2] / landmarks[:, 2:3]).reshape(m, 2 * keypoints)  # rescale
                landmarks = np.array([x * y + y - 1 for x, y in zip(landmarks, mask_landmarks)])
            else:  # affine
                landmarks = landmarks[:, :2].reshape(m, 2 * keypoints)
            targets[:, -2 * keypoints:] = landmarks
    return img, targets, seg


def _mirror(image, boxes, landmarks, segs, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob and len(landmarks) == 0:  # 关键点检测不支持翻转
        image = image[:, ::-1]
        segs = segs[:, ::-1] if len(segs) else np.array([])
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # add for 5 landmarks  使用镜像的时候要注意对关键点进行重新排位
        # if len(landmarks) > 0:
        #     landmarks[:, 0::2] = width - landmarks[:, 2::-2]
    return image, boxes, landmarks, segs


def preproc(img, input_size, seg_target, swap=(2, 0, 1)):
    h, w, c = img.shape
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], c), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / h, input_size[1] / w)
    resized_img = cv2.resize(
        img,
        (int(w * r), int(h * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(h * r), : int(w * r)] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    if len(seg_target) > 0:
        h, w, segcls = seg_target.shape
        padded_seg = np.zeros((int(input_size[0]), int(input_size[1]), segcls), dtype=np.uint8)
        resized_seg = cv2.resize(
            seg_target, (int(w * r), int(h * r)),
            interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        if len(resized_seg.shape) == 2:
            resized_seg = np.expand_dims(resized_seg, axis=-1)
        padded_seg[: int(h * r), : int(w * r)] = resized_seg
        padded_seg = padded_seg.transpose(swap)
        padded_seg = np.ascontiguousarray(padded_seg, dtype=np.float32)
    else:
        padded_seg = seg_target
    return padded_img, r, padded_seg


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, keypoints=0, segcls=0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.keypoints = keypoints
        self.segcls = segcls

    def __call__(self, image, targets, input_dim, seg_targets):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        landmarks = targets[:, -2 * self.keypoints:].copy() if self.keypoints > 0 else np.array([])
        seg = seg_targets.copy() if self.segcls > 0 else np.array([])
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5 + 2 * self.keypoints), dtype=np.float32)
            if self.keypoints > 0:
                targets[..., -2 * self.keypoints:] = targets[..., -2 * self.keypoints:] - 1
            if self.segcls > 0:
                h, w, _ = image.shape
                seg_targets = seg_targets * 0
            else:
                seg_targets = np.array([])
            image, r_o, seg_targets = preproc(image, input_dim, seg_targets)
            return image, targets, seg_targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        boxes_o = xyxy2cxcywh(boxes_o)
        landmarks_o = targets_o[:, -2 * self.keypoints:].copy() if self.keypoints > 0 else np.array([])
        segs_o = seg_targets.copy() if self.segcls > 0 else np.array([])

        if random.random() < self.hsv_prob:
            augment_hsv(image)

        image_t, boxes, landmarks, seg_t = _mirror(image, boxes, landmarks, seg, self.flip_prob)  # flip
        height, width, _ = image_t.shape
        image_t, r_, seg_t = preproc(image_t, input_dim, seg_t)  # to input size
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_
        landmarks *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        landmarks_t = landmarks[mask_b] if self.keypoints > 0 else np.array([])

        if len(boxes_t) == 0:
            image_t, r_o, segs_o = preproc(image_o, input_dim, segs_o)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            landmarks_o *= r_o
            landmarks_t = landmarks_o

        labels_t = np.expand_dims(labels_t, 1)

        if self.keypoints > 0:
            targets_t = np.hstack((labels_t, boxes_t, landmarks_t))
            padded_labels = np.zeros((self.max_labels, 5 + self.keypoints * 2))
        else:
            targets_t = np.hstack((labels_t, boxes_t))
            padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[:self.max_labels]] = targets_t[:self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels, seg_t


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _, seg = preproc(img, input_size, np.array([]), self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
