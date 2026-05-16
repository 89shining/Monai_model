import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DDUNet2D(nn.Module):
    """
    2D DDUNet for medical segmentation.
    Input shape example: (N, 1, 256, 256)
    Output shape: (N, 1, 256, 256), sigmoid probability map.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # 1) Multi-scale dilated branches
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=3, dilation=1, padding=1, bias=False)
        self.branch2 = nn.Conv2d(in_channels, 128, kernel_size=3, dilation=2, padding=2, bias=False)
        self.branch3 = nn.Conv2d(in_channels, 256, kernel_size=3, dilation=4, padding=4, bias=False)
        self.branch_relu = nn.ReLU(inplace=True)

        self.pool_b1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_b2 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.pool_b3 = nn.MaxPool2d(kernel_size=3, stride=8, padding=1)

        # 2) Encoder (5 levels)
        # level1
        self.enc1 = ConvBlock(1, 64)
        # level2 concat branch1 (64 + 64 -> 128 input)
        self.enc2 = ConvBlock(128, 128)
        # level3 concat branch2 (128 + 128 -> 256 input)
        self.enc3 = ConvBlock(256, 256)
        # level4 concat branch3 (256 + 256 -> 512 input)
        self.enc4 = ConvBlock(512, 512)
        # level5 bottleneck
        self.enc5 = ConvBlock(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3) Decoder
        self.dec4 = UpBlock(1024, 512, 512)
        self.dec3 = UpBlock(512, 256, 256)
        self.dec2 = UpBlock(256, 128, 128)
        self.dec1 = UpBlock(128, 64, 64)

        # 4) Output head
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # multi-scale branch features
        b1 = self.pool_b1(self.branch_relu(self.branch1(x)))
        b2 = self.pool_b2(self.branch_relu(self.branch2(x)))
        b3 = self.pool_b3(self.branch_relu(self.branch3(x)))

        # encoder level1
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        # encoder level2 + concat branch1
        e2_in = torch.cat([p1, b1], dim=1)
        e2 = self.enc2(e2_in)
        p2 = self.pool(e2)

        # encoder level3 + concat branch2
        e3_in = torch.cat([p2, b2], dim=1)
        e3 = self.enc3(e3_in)
        p3 = self.pool(e3)

        # encoder level4 + concat branch3
        e4_in = torch.cat([p3, b3], dim=1)
        e4 = self.enc4(e4_in)
        p4 = self.pool(e4)

        # encoder level5
        e5 = self.enc5(p4)

        # decoder
        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # output
        out = self.out_conv(d1)
        out = torch.sigmoid(out)
        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1).float()

        intersection = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target)


if __name__ == "__main__":
    # quick shape check
    model = DDUNet2D(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print("Input:", tuple(x.shape))
    print("Output:", tuple(y.shape))
