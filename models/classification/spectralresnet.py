import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.classification.sngpnorm import spectral_norm_conv, spectral_norm_fc, SpectralBatchNorm2d
import numpy as np
from torch.nn.utils.spectral_norm import spectral_norm

coeff = 3
n_power_iterations=1


def wrapped_bn(num_features):
    bn = SpectralBatchNorm2d(num_features, coeff)
    # bn = nn.BatchNorm2d(num_features)

    return bn


def wrapped_conv(in_c, out_c, kernel_size, stride):
    padding = 1 if kernel_size == 3 else 0

    conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

    if kernel_size == 1:
        # use spectral norm fc, because bound are tight for 1x1 convolutions
        wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
    else:
        # Otherwise use spectral norm conv, with loose bound
        # input_dim = (in_c, input_size, input_size)
        wrapped_conv = spectral_norm(conv)
        # wrapped_conv = spectral_norm_conv(
        #     conv, coeff, input_dim, n_power_iterations
        # )

    return wrapped_conv


def conv3x3(in_planes, out_planes, stride=1):
    out = wrapped_conv(in_planes, out_planes, kernel_size=3, stride=stride)
    return out


def conv2d_size(input_size, kern, pad, stride, dial=1):
    output_size = int(np.floor((input_size + 2 * pad - dial * (kern - 1) - 1) / stride + 1))
    return output_size


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0))
        self.bn1 = wrapped_bn(planes)

        self.conv2 = spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1))
        self.bn2 = wrapped_bn(planes)

        self.conv3 = spectral_norm(nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0))
        self.bn3 = wrapped_bn(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            sp_conv = wrapped_conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
            sp_bn = wrapped_bn(self.expansion*planes)
            self.shortcut = nn.Sequential(
                sp_conv,
                sp_bn
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = wrapped_bn(planes)
        # second_input = (input_size - 1) // stride + 1
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = wrapped_bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            sp_conv = wrapped_conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
            sp_bn = wrapped_bn(self.expansion*planes)
            self.shortcut = nn.Sequential(
                sp_conv,
                sp_bn
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpectralResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=None):
        super(SpectralResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(in_planes=3, out_planes=64)
        self.bn1 = wrapped_bn(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self._num_classes = num_classes
        if num_classes is not None:
            self.linear = nn.Linear(512*block.expansion, num_classes)

        self._block_expansion = block.expansion
        self.penultimate_layer = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_penultimate_dim(self):
        return 512 * self._block_expansion

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.penultimate_layer = out
        if self._num_classes is not None:
            out = self.linear(out)
        return out


def SpectralResNet18(num_classes: int = None):
    return SpectralResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def SpectralResNet82(num_classes: int = None):
    return SpectralResNet(BasicBlock, [5, 15, 15, 5], num_classes=num_classes)


def SpectralResNet101(num_classes: int = None):
    return SpectralResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)



