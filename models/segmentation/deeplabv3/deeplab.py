import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segmentation.deeplabv3.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.segmentation.deeplabv3.aspp import build_aspp
from models.segmentation.deeplabv3.decoder import build_decoder
from models.segmentation.deeplabv3.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet_18', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn:
            batchnorm = SynchronizedBatchNorm2d
        else:
            batchnorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, batchnorm)
        self.aspp = build_aspp(backbone, output_stride, batchnorm)
        self.decoder = build_decoder(num_classes, backbone, batchnorm)
        self.penultimate = None
        self.penultimate_dim = 256

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x, penultimate = self.decoder(x, low_level_feat)
        penultimate = F.interpolate(penultimate, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        self.penultimate = penultimate
        return x, None, penultimate

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class DeepLabRECON(nn.Module):
    def __init__(self, backbone='resnet_18', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabRECON, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn:
            batchnorm = SynchronizedBatchNorm2d
        else:
            batchnorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, batchnorm)
        self.aspp = build_aspp(backbone, output_stride, batchnorm)
        self.decoder = build_decoder(num_classes, backbone, batchnorm)
        self.recon = build_decoder(3, backbone, batchnorm)
        self.penultimate = None
        self.penultimate_dim = 256

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        recon = self.recon(x, low_level_feat)
        x, penultimate = self.decoder(x, low_level_feat)
        penultimate = F.interpolate(penultimate, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        recon = F.interpolate(recon, size=input.size()[2:], mode='bilinear', align_corners=True)

        self.penultimate = penultimate
        return x, recon, penultimate

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p