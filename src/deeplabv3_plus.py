import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ----------------------------
# ASPP MODULE
# ----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode="bilinear", align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x)


# ----------------------------
# DEEPLABV3+ MODEL
# ----------------------------
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Encoder
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1  # low-level features
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ASPP
        self.aspp = ASPP(2048, 256)

        # Decoder
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        x = self.layer0(x)
        low_level = self.layer1(x)

        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=low_level.shape[2:], mode="bilinear", align_corners=False)

        low_level = self.low_level_conv(low_level)
        x = torch.cat([x, low_level], dim=1)

        x = self.decoder(x)
        x = self.classifier(x)

        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


# ----------------------------
# SHAPE TEST
# ----------------------------
if __name__ == "__main__":
    model = DeepLabV3Plus()
    model.eval()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
