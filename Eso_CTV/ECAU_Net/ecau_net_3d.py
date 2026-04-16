import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECABlock3D(nn.Module):
    """Efficient Channel Attention adapted to 3D feature maps."""

    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # [B, C, 1, 1, 1]
        y = y.squeeze(-1).squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv1d(y)
        y = self.act(y).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1, 1]
        return x * y.expand_as(x)


class CrossStageAttention3D(nn.Module):
    """
    Cross-stage attention fusion:
    - gathers multi-stage encoder features,
    - aligns them to decoder scale,
    - produces attention-guided fused skip feature.
    """

    def __init__(self, decoder_channels: int, context_channels: list[int]):
        super().__init__()
        self.context_proj = nn.ModuleList(
            [nn.Conv3d(ch, decoder_channels, kernel_size=1, bias=False) for ch in context_channels]
        )
        self.context_norm = nn.ModuleList([nn.InstanceNorm3d(decoder_channels, affine=True) for _ in context_channels])

        self.query_proj = nn.Sequential(
            nn.Conv3d(decoder_channels, decoder_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(decoder_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.spatial_att = nn.Sequential(
            nn.Conv3d(decoder_channels * 2, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(decoder_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(decoder_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.eca = ECABlock3D(decoder_channels, k_size=3)

    def forward(self, query: torch.Tensor, contexts: list[torch.Tensor]) -> torch.Tensor:
        target_size = query.shape[2:]

        merged = 0.0
        for feat, proj, norm in zip(contexts, self.context_proj, self.context_norm):
            feat = proj(feat)
            feat = norm(feat)
            feat = F.interpolate(feat, size=target_size, mode="trilinear", align_corners=False)
            merged = merged + feat

        q = self.query_proj(query)
        att = self.spatial_att(torch.cat([q, merged], dim=1))
        fused = merged * att + q
        return self.eca(fused)


class ECAUNet3D(nn.Module):
    """
    3D ECAU-Net style architecture:
    U-Net backbone + cross-stage attention skip fusion + channel attention.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, channels: tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()
        if len(channels) != 5:
            raise ValueError("channels must be a 5-length tuple, e.g. (32, 64, 128, 256, 512)")

        c1, c2, c3, c4, c5 = channels

        self.enc1 = ConvBlock3D(in_channels, c1)
        self.down1 = nn.Conv3d(c1, c2, kernel_size=2, stride=2, bias=False)

        self.enc2 = ConvBlock3D(c2, c2)
        self.down2 = nn.Conv3d(c2, c3, kernel_size=2, stride=2, bias=False)

        self.enc3 = ConvBlock3D(c3, c3)
        self.down3 = nn.Conv3d(c3, c4, kernel_size=2, stride=2, bias=False)

        self.enc4 = ConvBlock3D(c4, c4)
        self.down4 = nn.Conv3d(c4, c5, kernel_size=2, stride=2, bias=False)

        self.bottleneck = ConvBlock3D(c5, c5)

        self.up4 = nn.ConvTranspose3d(c5, c4, kernel_size=2, stride=2)
        self.csa4 = CrossStageAttention3D(c4, [c4, c3])
        self.dec4 = ConvBlock3D(c4 + c4, c4)

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.csa3 = CrossStageAttention3D(c3, [c3, c2])
        self.dec3 = ConvBlock3D(c3 + c3, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.csa2 = CrossStageAttention3D(c2, [c2, c1])
        self.dec2 = ConvBlock3D(c2 + c2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.csa1 = CrossStageAttention3D(c1, [c1])
        self.dec1 = ConvBlock3D(c1 + c1, c1)

        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _match_shape(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

        d4 = self._match_shape(self.up4(b), e4)
        s4 = self.csa4(d4, [e4, e3])
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self._match_shape(self.up3(d4), e3)
        s3 = self.csa3(d3, [e3, e2])
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self._match_shape(self.up2(d3), e2)
        s2 = self.csa2(d2, [e2, e1])
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self._match_shape(self.up1(d2), e1)
        s1 = self.csa1(d1, [e1])
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        return self.head(d1)
