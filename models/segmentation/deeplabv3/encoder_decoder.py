import torch.nn as nn
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.functional import interpolate


class DownSampleBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=[1, 1], kernelSize=[3, 3], poolSz=[2, 2], poolStride=[2, 2], padding=[1, 1], downSample=False):
        super(DownSampleBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.kernelSize = kernelSize
        self.pool_sz = poolSz
        self.pool_stride = poolStride
        self.downSample = downSample
        self.conv1  = nn.Conv2d(self.inplanes, self.planes, kernel_size=self.kernelSize[0], stride=self.stride[0], padding=padding[0], bias=False)
        self.bn1    = nn.BatchNorm2d(self.planes, eps=1e-05, momentum=0.1, affine=True)
        self.relu1  = nn.ReLU(inplace=True)
        self.conv2  = nn.Conv2d(self.planes, self.planes, kernel_size=self.kernelSize[1], stride=self.stride[1], padding=padding[1], bias=False)
        self.bn2    = nn.BatchNorm2d(self.planes, eps=1e-05, momentum=0.1, affine=True)
        self.relu2  = nn.ReLU(inplace=True)
        if self.downSample is True:
            self.convds = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.pool_sz, stride=self.pool_stride, return_indices=False, ceil_mode=False)

    def forward(self, *input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.downSample is True:
            x = self.convds(x)
        x = self.maxpool1(x)
        return x

class Encoder(nn.Module):

    def __init__(self, outStride, stride, kernelSize):
        super(Encoder, self).__init__()
        self.stride = stride
        self.kernelSize = kernelSize
        self.outStride = outStride
        self.featureBlock = 64
        self.poolSz = [2, 2]
        self.poolStride = [2, 2]
        self.layers = nn.Module
        self.layers.add_module(self, name='conv0', module=DownSampleBlock(inplanes=1,
                                                                    planes=self.featureBlock,
                                                                    stride=[2, 1],
                                                                    kernelSize=[7, 1],
                                                                    poolSz=[2, 2],
                                                                    poolStride=[2, 2],
                                                                    downSample=False,
                                                                    padding=[3, 1]))
        loop_times = torch.floor(torch.log2(torch.tensor(self.outStride).float())).item()
        for i in range(1, int(loop_times)-1):
            self.layers.add_module(self, name='conv{}'.format(i), module=DownSampleBlock(inplanes=i*self.featureBlock,
                                                                                   planes=2*i*self.featureBlock,
                                                                                   stride=self.stride,
                                                                                   kernelSize=self.kernelSize,
                                                                                   poolSz=self.poolSz,
                                                                                   poolStride=self.poolStride))
    def forward(self, x):
        x = self.layers(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_classes, outStride, featureBlock=64):
        super(Decoder, self).__init__()
        self.out_stride = outStride
        self.featureBlock = featureBlock
        loopTimes = torch.floor(torch.log2(torch.tensor(self.featureBlock).float())).item()
        lowLevelInplanes = int(loopTimes * featureBlock)
        self.conv1 = nn.Conv2d(lowLevelInplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, lowLevelFeat):
        lowLevelFeat = self.conv1(lowLevelFeat)
        lowLevelFeat = self.bn1(lowLevelFeat)
        lowLevelFeat = self.relu(lowLevelFeat)

        x = F.interpolate(lowLevelFeat, size=lowLevelFeat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, lowLevelFeat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    outStride = 8
    stride = [1, 1]
    kernelSize = [3, 3]
    print(Decoder(num_classes=4, outStride=outStride, featureBlock=64))
