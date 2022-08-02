import torch
import torch.nn as nn

from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOv7_Backone(nn.Module):
    def __init__(self, img_channel=3):
        super().__init__()
        self.conv1 = BaseConv(img_channel, 32, 3, 2, act='lrelu')
        self.conv2 = BaseConv(32, 64, 3, 2, act='lrelu')
        self.conv3 = BaseConv(64, 32, 1, 1, act='lrelu')
        self.conv4 = BaseConv(64, 32, 1, 1, act='lrelu')
        self.conv5 = BaseConv(32, 32, 3, 1, act='lrelu')
        self.conv6 = BaseConv(32, 32, 3, 1, act='lrelu')
        self.conv7 = BaseConv(32*4, 64, 1, 1, act='lrelu')

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv9 = BaseConv(64, 64, 1, 1, act='lrelu')
        self.conv10 = BaseConv(64, 64, 1, 1, act='lrelu')
        self.conv11 = BaseConv(64, 64, 3, 1, act='lrelu')
        self.conv12 = BaseConv(64, 64, 3, 1, act='lrelu')
        self.conv14 = BaseConv(64*4, 128, 1, 1, act='lrelu')

        self.conv16 = BaseConv(128, 128, 1, 1, act='lrelu')
        self.conv17 = BaseConv(128, 128, 1, 1, act='lrelu')
        self.conv18 = BaseConv(128, 128, 3, 1, act='lrelu')
        self.conv19 = BaseConv(128, 128, 3, 1, act='lrelu')
        self.conv21 = BaseConv(128*4, 256, 1, 1, act='lrelu')

        self.conv23 = BaseConv(256, 256, 1, 1, act='lrelu')
        self.conv24 = BaseConv(256, 256, 1, 1, act='lrelu')
        self.conv25 = BaseConv(256, 256, 3, 1, act='lrelu')
        self.conv26 = BaseConv(256, 256, 3, 1, act='lrelu')
        self.conv28 = BaseConv(256*4, 512, 1, 1, act='lrelu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x1 = self.conv3(x)
        x2 = self.conv4(x)
        x3 = self.conv5(x2)
        x4 = self.conv6(x3)
        x4 = torch.cat([x4, x3, x2, x1], dim=1)
        x7 = self.conv7(x4)

        x7 = self.mp(x7)
        x9 = self.conv9(x7)
        x10 = self.conv10(x7)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x12 = torch.cat([x12, x11, x10, x9], dim=1)
        x14 = self.conv14(x12)

        x15 = self.mp(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x15)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x19 = torch.cat([x19, x18, x17, x16], dim=1)
        x21 = self.conv21(x19)

        x22 = self.mp(x21)
        x23 = self.conv23(x22)
        x24 = self.conv24(x22)
        x25 = self.conv25(x24)
        x26 = self.conv26(x25)
        x26 = torch.cat([x26, x25, x24, x23], dim=1)
        x28 = self.conv28(x26)
        return [x28, x21, x14]


class YOLO7TINY(nn.Module):
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
        self.backbone = YOLOv7_Backone(img_channel=img_channel)
        self.conv29 = BaseConv(512, 256, 1, 1, act='lrelu')
        self.conv30 = BaseConv(512, 256, 1, 1, act='lrelu')
        self.conv35 = BaseConv(256*4, 256, 1, 1, act='lrelu')
        self.conv37 = BaseConv(256*2, 256, 1, 1, act='lrelu')
        self.upsample = nn.Upsample(None, 2, 'nearest')
        self.conv38 = BaseConv(256, 128, 1, 1, act='lrelu')
        self.conv40 = BaseConv(256, 128, 1, 1, act='lrelu')

        self.conv42 = BaseConv(128*2, 64, 1, 1, act='lrelu')
        self.conv43 = BaseConv(128*2, 64, 1, 1, act='lrelu')
        self.conv44 = BaseConv(64, 64, 3, 1, act='lrelu')
        self.conv45 = BaseConv(64, 64, 3, 1, act='lrelu')
        self.conv47 = BaseConv(64*4, 128, 1, 1, act='lrelu')

        self.conv48 = BaseConv(128, 64, 1, 1, act='lrelu')
        self.conv50 = BaseConv(64*2, 64, 1, 1, act='lrelu')

        self.conv52 = BaseConv(64*2, 32, 1, 1, act='lrelu')
        self.conv53 = BaseConv(64*2, 32, 1, 1, act='lrelu')
        self.conv54 = BaseConv(32, 32, 3, 1, act='lrelu')
        self.conv55 = BaseConv(32, 32, 3, 1, act='lrelu')
        self.conv57 = BaseConv(32*4, 64, 1, 1, act='lrelu')

        self.conv58 = BaseConv(64, 128, 3, 2, act='lrelu')

        self.conv60 = BaseConv(128*2, 64, 1, 1, act='lrelu')
        self.conv61 = BaseConv(128*2, 64, 1, 1, act='lrelu')
        self.conv62 = BaseConv(64, 64, 3, 1, act='lrelu')
        self.conv63 = BaseConv(64, 64, 3, 1, act='lrelu')
        self.conv65 = BaseConv(64*4, 128, 1, 1, act='lrelu')

        self.conv66 = BaseConv(128, 256, 3, 2, act='lrelu')

        self.conv68 = BaseConv(256*2, 128, 1, 1, act='lrelu')
        self.conv69 = BaseConv(256*2, 128, 1, 1, act='lrelu')
        self.conv70 = BaseConv(128, 128, 3, 1, act='lrelu')
        self.conv71 = BaseConv(128, 128, 3, 1, act='lrelu')
        self.conv73 = BaseConv(128*4, 256, 1, 1, act='lrelu')

        self.conv74 = BaseConv(64, 128, 3, 1, act='lrelu')
        self.conv75 = BaseConv(128, 256, 3, 1, act='lrelu')
        self.conv76 = BaseConv(256, 512, 3, 1, act='lrelu')

    def forward(self, x):
        P5, P4, P3 = self.backbone(x)
        x29 = self.conv29(P5)
        x30 = self.conv30(P5)
        x31 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)(x30)
        x32 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)(x30)
        x33 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)(x30)
        x34 = torch.cat([x33, x32, x31, x30], dim=1)
        x35 = self.conv35(x34)
        x36 = torch.cat([x35, x29], dim=1)
        x37 = self.conv37(x36)

        x38 = self.conv38(x37)
        x39 = self.upsample(x38)
        x40 = self.conv40(P4)
        x41 = torch.cat([x40, x39], dim=1)

        x42 = self.conv42(x41)
        x43 = self.conv43(x41)
        x44 = self.conv44(x43)
        x45 = self.conv45(x44)
        x46 = torch.cat([x45, x44, x43, x42], dim=1)
        x47 = self.conv47(x46)

        x48 = self.conv48(x47)
        x49 = self.upsample(x48)
        x50 = self.conv50(P3)
        x51 = torch.cat([x50, x49], dim=1)

        x52 = self.conv52(x51)
        x53 = self.conv53(x51)
        x54 = self.conv54(x53)
        x55 = self.conv55(x54)
        x56 = torch.cat([x55, x54, x53, x52], dim=1)
        x57 = self.conv57(x56)

        x58 = self.conv58(x57)
        x59 = torch.cat([x58, x47], dim=1)

        x60 = self.conv60(x59)
        x61 = self.conv61(x59)
        x62 = self.conv62(x61)
        x63 = self.conv63(x62)
        x64 = torch.cat([x63, x62, x61, x60], dim=1)
        x65 = self.conv65(x64)

        x66 = self.conv66(x65)
        x67 = torch.cat([x66, x37], dim=1)

        x68 = self.conv68(x67)
        x69 = self.conv69(x67)
        x70 = self.conv70(x69)
        x71 = self.conv71(x70)
        x72 = torch.cat([x71, x70, x69, x68], dim=1)
        x73 = self.conv73(x72)

        x74 = self.conv74(x57)
        x75 = self.conv75(x65)
        x76 = self.conv76(x73)

        return x74, x75, x76

if __name__ == '__main__':
    model = YOLO7TINY(img_channel=3)
    img = torch.rand(1, 3, 640, 640)
    y = model(img)
    for item in y:
        print(item.shape)
    # torch.Size([1, 128, 80, 80])
    # torch.Size([1, 256, 40, 40])
    # torch.Size([1, 512, 20, 20])
