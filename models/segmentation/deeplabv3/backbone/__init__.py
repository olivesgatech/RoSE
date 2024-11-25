from models.segmentation.deeplabv3.backbone import resnet, resnet2


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet_101':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet_18':
        return resnet2.resnet18(pretrained=False, progress=True)
    elif backbone == 'resnet_34':
        return resnet2.resnet34(pretrained=False, progress=True)
    elif backbone == 'resnet_50':
        return resnet2.resnet34(pretrained=False, progress=True)
    else:
        raise NotImplementedError
