import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, in_chans=3, num_classes=1000, activation=nn.GELU,
                 dropout=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        if num_classes > 0:
            # self.head = nn.Linear(dim, num_classes)
            self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(dim, num_classes),
        ) if dropout else nn.Linear(dim, num_classes)
        else:
            self.head = nn.Identity()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            activation(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    activation(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                activation(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pooling(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

def convmixer_768_32(num_classes: int, pretrained=False, dropout=False):
    model_path = '/home/ryan/models/convmixer_768_32_ks7_p7_relu.pth.tar'
    mixer = ConvMixer(dim=768, depth=32, kernel_size=7, patch_size=7, activation=nn.ReLU)
    mixer.load_state_dict(torch.load(model_path))
    mixer.stem = nn.Sequential(
            nn.Conv2d(3, 768, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(768)
        )
    # mixer.head = nn.Linear(768, num_classes)
    mixer.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(768, num_classes),
            # nn.BatchNorm1d(num_classes),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),
            # nn.Linear(num_classes, num_classes)
        ) if dropout else nn.Linear(768, num_classes)
    return mixer

def convmixer32(num_classes: int, dropout: bool = False):
    dim = 256
    out = ConvMixer(num_classes=num_classes, dim=dim, kernel_size=8, depth=8, patch_size=1, dropout=dropout)
    return out
