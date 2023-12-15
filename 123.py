import torch, time
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, CSPDarknet, YOLOPAFPNSLIM, YOLO7TINY


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # in_channels = [128, 256, 512, 1024]
    # in_features = ("dark2", "dark3", "dark4", "dark5")
    in_channels = [256, 512, 1024]
    in_features = ("dark3", "dark4", "dark5")
    depth = 0.33
    width = 0.50
    img_channel = 3

    # backbone = CSPDarknet(img_channel, depth, width, depthwise=False, act="silu", out_features=in_features)

    # backbone = YOLOPAFPN(img_channel, depth, width, in_channels=in_channels, in_features=in_features,
    #                      backbone_name='CSPDarknet').cuda()  # CSPDarknet  CoAtNet  0.01363 s

    # backbone = YOLOPAFPNSLIM(img_channel, depth, width, in_channels=in_channels, in_features=in_features,
    #                      backbone_name='CSPDarknet').cuda()  # CSPDarknet  CoAtNet  0.01215 s

    backbone = YOLO7TINY(img_channel=img_channel, width=width, act='lrelu').cuda()  # 0.00808 s  0.00644 s

    ## 输入320*320  输出128,40,40  256,20,20  512,10,10        seg多一个 64,80,80

    head = YOLOXHead(1, width, in_channels=in_channels, keypoints=0, segcls=0,
                     act="silu", repeat=0).cuda()  # 0.01101 s   0.00827 s
    model = YOLOX(backbone, head).cuda()
    model.fuse()
    model.eval()
    img = torch.randn(10, 3, 192, 320).cuda()
    out, seg_out = model(img)
    # print(out.shape, seg_out.shape)
    # print('???', seg_out.shape)
    #
    # torch.onnx.export(backbone, img, 'model2.onnx', opset_version=12,
    #                       input_names=['images'], output_names=['output'],
    #                       dynamic_axes={'images': {0: 'batch_size', 2: 'h', 3: 'w'},  # , 2: 'h', 3: 'w'
    #                                     'output': {0: 'batch_size'},
    #                                     }
    #                       )

    # outs = backbone(img)
    # print(count_parameters(backbone))  # 5990464  1891344
    # # features = [outs[f] for f in in_features]
    # for out in outs:
    #     print(out.shape)

    for i in range(50):
        out, seg_out = model(img)
        # backbone(img)

    start_time = time.perf_counter()
    # backbone(img)
    out, seg_out = model(img)
    end_time = time.perf_counter()
    print(f'time cost: {round(end_time - start_time, 5)} s')
