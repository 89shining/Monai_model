import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DDCBlock(nn.Module):
    """
    Deep dilated convolution block used to aggregate multi-scale context.
    Branch dilations follow the paper setting: 1, 2, 4.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.branch_d1 = ConvBNReLU(channels, channels, dilation=1)
        self.branch_d2 = ConvBNReLU(channels, channels, dilation=2)
        self.branch_d4 = ConvBNReLU(channels, channels, dilation=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_d1(x)
        b2 = self.branch_d2(x)
        b4 = self.branch_d4(x)
        merged = torch.cat([b1, b2, b4], dim=1)
        return self.fuse(merged)


class DDUnet(nn.Module):
    """
    Paper-style DDUnet approximation:
    - 5-level U-Net encoder/decoder
    - DDC block fused on each encoder skip feature
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()

        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bottom = DoubleConv(c4, c5)

        self.ddc1 = DDCBlock(c1)
        self.ddc2 = DDCBlock(c2)
        self.ddc3 = DDCBlock(c3)
        self.ddc4 = DDCBlock(c4)

        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _pad_if_needed(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        diff_y = ref.size(2) - x.size(2)
        diff_x = ref.size(3) - x.size(3)
        if diff_x == 0 and diff_y == 0:
            return x
        return nn.functional.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottom(self.pool(e4))

        s1 = e1 + self.ddc1(e1)
        s2 = e2 + self.ddc2(e2)
        s3 = e3 + self.ddc3(e3)
        s4 = e4 + self.ddc4(e4)

        d4 = self.up4(b)
        d4 = self._pad_if_needed(d4, s4)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self.up3(d4)
        d3 = self._pad_if_needed(d3, s3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        d2 = self._pad_if_needed(d2, s2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        d1 = self._pad_if_needed(d1, s1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.head(d1)
