import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive 3x3 conv blocks with BN and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    """Upsample by 2 then apply a 3x3 conv block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionBlock(nn.Module):
    """Attention gate for U-Net skip connections."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_proj = self.w_g(g)
        x_proj = self.w_x(x)
        psi = self.relu(g_proj + x_proj)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net implementation extracted from sfczekalski/attention_unet."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        filters = [64, 128, 256, 512, 1024]

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(in_channels, filters[0])
        self.conv2 = ConvBlock(filters[0], filters[1])
        self.conv3 = ConvBlock(filters[1], filters[2])
        self.conv4 = ConvBlock(filters[2], filters[3])
        self.conv5 = ConvBlock(filters[3], filters[4])

        self.up5 = UpConv(filters[4], filters[3])
        self.att5 = AttentionBlock(filters[3], filters[3], filters[2])
        self.up_conv5 = ConvBlock(filters[4], filters[3])

        self.up4 = UpConv(filters[3], filters[2])
        self.att4 = AttentionBlock(filters[2], filters[2], filters[1])
        self.up_conv4 = ConvBlock(filters[3], filters[2])

        self.up3 = UpConv(filters[2], filters[1])
        self.att3 = AttentionBlock(filters[1], filters[1], filters[0])
        self.up_conv3 = ConvBlock(filters[2], filters[1])

        self.up2 = UpConv(filters[1], filters[0])
        self.att2 = AttentionBlock(filters[0], filters[0], 32)
        self.up_conv2 = ConvBlock(filters[1], filters[0])

        self.out_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        e1 = self.conv1(x)
        e2 = self.conv2(self.maxpool(e1))
        e3 = self.conv3(self.maxpool(e2))
        e4 = self.conv4(self.maxpool(e3))
        e5 = self.conv5(self.maxpool(e4))

        # decoder + attention-gated skip connections
        d5 = self.up5(e5)
        e4_att = self.att5(g=d5, x=e4)
        d5 = self.up_conv5(torch.cat((e4_att, d5), dim=1))

        d4 = self.up4(d5)
        e3_att = self.att4(g=d4, x=e3)
        d4 = self.up_conv4(torch.cat((e3_att, d4), dim=1))

        d3 = self.up3(d4)
        e2_att = self.att3(g=d3, x=e2)
        d3 = self.up_conv3(torch.cat((e2_att, d3), dim=1))

        d2 = self.up2(d3)
        e1_att = self.att2(g=d2, x=e1)
        d2 = self.up_conv2(torch.cat((e1_att, d2), dim=1))

        return self.out_conv(d2)
