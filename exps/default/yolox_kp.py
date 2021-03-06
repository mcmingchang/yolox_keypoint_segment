#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        #### s
        self.depth = 0.33
        self.width = 0.50
        # #### m
        # self.depth = 0.67
        # self.width = 0.75
        #### l
        # self.depth = 1.0
        # self.width = 1.0
        #### x
        # self.depth = 1.33
        # self.width = 1.25

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.keypoints = 4
        self.num_classes = 2
        self.data_num_workers = 0
        self.pin_memory = False
        self.input_size = (320, 320)  # (height, width)
        self.test_size = (320, 320)
        self.data_dir = 'datasets/plate_kp'
        self.mosaic_prob = 1
        self.multiscale_range = 5
        self.adam = True

        # self.backbone_name = 'CoAtNet'
        # if self.backbone_name == 'CoAtNet':
        #     self.multiscale_range = 0
