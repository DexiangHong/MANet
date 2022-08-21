from modeling.deeplab.backbone import resnet, mobilenet
import torch


def build_backbone(backbone, output_stride, BatchNorm, pretrained_path=None):
    if backbone == 'resnet':
        model = resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'mobilenet':
        model = mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError

    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    return model


if __name__ == '__main__':
    from torch import nn
    model = build_backbone('resnet', 32, nn.BatchNorm2d)
    x = torch.randn([2, 3, 320, 320])
    outs = model(x, True)
    for out in outs:
        print(out.shape)