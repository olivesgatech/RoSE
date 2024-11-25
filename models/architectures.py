from models.classification.mlp import MLP
from models.classification.resnet import ResNet18, ResNet34, ResNetn
from models.classification.spectralresnetv2 import SpectralResNet18
from models.classification.sdnresnet import ResNetSDN18
from models.classification.torchmodels import *
from models.classification.convmixer import convmixer_768_32, convmixer32
from models.classification.vitpt import vit_model_l_16
from models.segmentation.deeplabv3.deeplab import DeepLab, DeepLabRECON
from data.datasets.shared.utils import DatasetStructure
from torchvision.models import resnet18

from config import BaseConfig
import torchvision.models as torch_models


def build_architecture(architecture: str, data_cfg: DatasetStructure, cfg: BaseConfig):
    num_sdn = -1
    if architecture == 'MLP':
        dim = (data_cfg.img_size ** 2)*3
        out = MLP(num_classes=data_cfg.num_classes, dim=dim)
    elif architecture == 'resnet-18':
        if cfg.data.dataset in ['XRAY', 'OCT']:
            out = ResNetn(type=18, num_classes=data_cfg.num_classes)
        else:
            # out = resnet18(pretrained=True, num_classes=200)
            out = ResNetn(type=18, num_classes=data_cfg.num_classes, pretrained=True)
    elif architecture == 'd-resnet-18':
        out = ResNetn(type=18, num_classes=data_cfg.num_classes, pretrained=True, dropout=True)
    elif architecture == 'resnet-34':
        out = ResNet34(num_classes=data_cfg.num_classes)
    elif architecture == 'spectral-resnet-18':
        out = SpectralResNet18(num_classes=data_cfg.num_classes)
    elif architecture == 'sdnresnet-18':
        out, num_sdn = ResNetSDN18(num_classes=data_cfg.num_classes)
    elif architecture == 'densenet-121':
        out = DenseNet(type=121, num_classes=data_cfg.num_classes)
    elif architecture == 'vgg-11':
        out = VGG(type=11, num_classes=data_cfg.num_classes)
    elif architecture == 'vgg-16':
        out = VGG(type=16, num_classes=data_cfg.num_classes)
    elif architecture == 'vit-l-16':
        print('VIT currently only implemented with imagenet')
        out = vit_model_l_16(data_cfg=data_cfg)
    elif architecture == 'vit':
        print('Using vit')
        import timm
        # out = timm.create_model("vit_base_patch16_384", pretrained=True)
        # out.head = nn.Linear(out.head.in_features, data_cfg.num_classes)
        out = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=data_cfg.num_classes)
        out.head = nn.Linear(out.head.in_features, data_cfg.num_classes)
    elif architecture == 'convmixer_dep':
        print('Using conv mixer')
        import timm
        # out = timm.create_model("vit_base_patch16_384", pretrained=True)
        # out.head = nn.Linear(out.head.in_features, data_cfg.num_classes)
        out = timm.create_model('convit_base', pretrained=True, num_classes=data_cfg.num_classes)
        out.patch_embed.proj = nn.Conv2d(3, out.patch_embed.proj.out_channels,
                                           kernel_size=3, stride=1, padding=1, bias=False)
    elif architecture == 'convmixer':
        print('Using conv mixer c10')
        # out = convmixer_768_32(pretrained=True, num_classes=data_cfg.num_classes)
        out = convmixer32(num_classes=data_cfg.num_classes)
    elif architecture == 'd-convmixer':
        print('Using dropout convmixer')
        # out = convmixer_768_32(pretrained=True, num_classes=data_cfg.num_classes)
        out = convmixer32(num_classes=data_cfg.num_classes, dropout=True)
    else:
        raise Exception('Architecture not implemented yet')

    return out, num_sdn


def build_segmentation(architecture: str, data_cfg: DatasetStructure):
    if architecture == 'deeplab-v3-rn18':
        return DeepLab(num_classes=data_cfg.num_classes, backbone='resnet_18')
    elif architecture == 'deeplab-v3-rn101':
        return DeepLab(num_classes=data_cfg.num_classes, backbone='resnet_101')
    elif architecture == 'deeplab-v3-recon':
        return DeepLabRECON(num_classes=data_cfg.num_classes)
    else:
        raise Exception('Architecture not implemented yet')
