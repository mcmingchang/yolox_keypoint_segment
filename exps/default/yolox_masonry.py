#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        #### s
        #self.depth = 0.33
        #self.width = 0.50
        # #### m
        self.depth = 0.67
        self.width = 0.75
        #### l
        # self.depth = 1.0
        # self.width = 1.0
        #### x
        # self.depth = 1.33
        # self.width = 1.25

        self.adam = True
        self.enable_mixup = False  # seg中只能为False
        self.multiscale_range = 0  # 随机变化的尺度 320:5  32*5~32*15
        self.hsv_prob = 0
        self.mosaic_scale = (0.1, 2)
        self.in_channels = [128, 256, 512, 1024]
        self.in_features = ("dark2", "dark3", "dark4", "dark5")
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_num_workers = 0
        self.pin_memory = True
        self.mosaic_prob = 0
        self.img_channel = 4
        self.num_classes = 7
        self.segcls = self.num_classes + 1
        self.input_size = (320, 320)  # (height, width)
        self.test_size = (320, 320)  # D:/train_model/YOLOX-main/
        self.data_dir = 'datasets/wall_seg/dataset'
        self.random_dataset = {
            'cate_ls': ['door', 'window', 'wire-box', 'brick-wall', 'electric-box', 'wallboard', 'tilt-brick'],
            'mask_order': ['brick-wall', 'wallboard', 'tilt-brick', 'door', 'window', 'electric-box', 'wire-box'],
            'cate_id': {'door': 10, 'window': 20, 'wire-box': 30, 'brick-wall': 40, 'electric-box': 50, 'wallboard': 60,
                        'tilt-brick': 70},
            'total': 30000}

        self.backbone_name = 'CoAtNet'
        if self.backbone_name == 'CoAtNet':
            self.multiscale_range = 0
