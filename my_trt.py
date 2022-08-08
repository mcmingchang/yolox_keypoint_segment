from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, CSPDarknet, YOLOPAFPNSLIM, YOLO7TINY
import tensorrt as trt
import torch
from torch import nn
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module
from torch2trt import torch2trt
from exps.default.yolox_ccpd import Exp


@torch.no_grad()
def main():
    exp = Exp()
    in_channels = [256, 512, 1024]
    img_channel = 3
    max_batch_size = 10
    backbone = YOLO7TINY(width=exp.width, img_channel=img_channel, act=exp.act).cuda()
    head = YOLOXHead(exp.num_classes, exp.width, in_channels=in_channels, keypoints=exp.keypoints,
                     act=exp.act, repeat=exp.repeat).cuda()

    model = YOLOX(backbone, head).cuda()

    ckpt = torch.load('last_epoch_ckpt-v7-lrelu.pth', map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    if exp.model_name == 'yolov7_tiny':
        model.fuse()
    model.eval()
    model.cuda()
    model.head.decode_in_inference = False
    x = torch.ones(1, img_channel, 192, 320).cuda()

    ######### trt
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << 32),
        max_batch_size=max_batch_size,
    )
    torch.save(model_trt.state_dict(), "model_trt.pth")

    ######## onnx
    # model = replace_module(model, nn.SiLU, SiLU)
    # dynamic = True
    # torch.onnx.export(
    #     model.cpu(),
    #     x.cpu(),
    #     'yolox.onnx',
    #     input_names=['images'],
    #     output_names=['output'],
    #     dynamic_axes={'images': {0: 'batch'},
    #                   'output': {0: 'batch'}} if dynamic else None,
    #     opset_version=None,
    # )

if __name__ == '__main__':
    main()