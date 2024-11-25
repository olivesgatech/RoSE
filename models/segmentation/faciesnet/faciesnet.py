import torch.nn as nn

# Setting up the network architecture


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


def double_conv_up(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, padding=0, dilation=1, kernel_size=3, stride=2, output_padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class FaciesSegNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(1, 3)
        self.dconv_down2 = double_conv(3, 10)
        self.dconv_down3 = double_conv(10, 30)
        self.dconv_down4 = double_conv(30, 40)
        self.dconv_down5 = double_conv(40, 60)

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up4 = double_conv_up(60, 50)
        self.dconv_up3 = double_conv_up(50, 30)
        self.dconv_up2 = double_conv_up(30, 15)
        self.dconv_up1 = double_conv_up(15, 8)

        self.conv_last = nn.Conv2d(8, n_class, 1)
        self.conv_reconstruct = nn.Conv2d(8, 1, 1)

    def forward(self, section):
        conv1 = self.dconv_down1(section)  # size is LxL
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)  # size is L/2 x L/2
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)  # size is L/4 x L/4
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)  # size is L/8 x L/8
        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)  # size is L/16 x L/16

        x = self.dconv_up4(conv5)

        x = self.dconv_up3(x)

        x = self.dconv_up2(x)

        x = self.dconv_up1(x)

        out = self.conv_last(x)[:, :, :section.shape[2], :section.shape[3]]
        reconstruct = self.conv_reconstruct(x)[:, :, :section.shape[2], :section.shape[3]]

        return out, reconstruct
