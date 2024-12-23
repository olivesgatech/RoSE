import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.classification.dropout import Dropout1d


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
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
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)


class ResNetn(nn.Module):
    def __init__(self, type=18, num_classes=10, pretrained=False, dropout: bool = False):
        super(ResNetn, self).__init__()
        if type == 18:
            self.backbone = models.resnet18(pretrained=pretrained)
        elif type == 34:
            self.backbone = models.resnet34(pretrained=pretrained)
        elif type == 50:
            self.backbone = models.resnet50(pretrained=pretrained)
        elif type == 101:
            self.backbone = models.resnet101(pretrained=pretrained)
        elif type == 152:
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError
        self.linear = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1000, num_classes),
            # nn.BatchNorm1d(num_classes),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),
            # nn.Linear(num_classes, num_classes)
        ) if dropout else nn.Linear(1000, num_classes)
        self.penultimate_layer = None

    def get_penultimate_dim(self):
        return 1000

    def forward(self, x):
        bbone = self.backbone(x)
        self.penultimate_layer = bbone
        out = self.linear(bbone)
        return out