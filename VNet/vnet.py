import torch
import torch.nn as nn
import torch.nn.functional as F


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    """BatchNorm3d variant that validates 5D inputs."""

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )


class ELUCons(nn.Module):
    def __init__(self, nchan, elu):
        super().__init__()
        self.activation = nn.ELU(inplace=True) if elu else nn.PReLU(nchan)

    def forward(self, x):
        return self.activation(x)


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super().__init__()
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)
        self.relu1 = ELUCons(nchan, elu)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super().__init__()
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(outChans, elu)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super().__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = nn.Dropout3d() if dropout else nn.Identity()
        self.relu1 = ELUCons(outChans, elu)
        self.relu2 = ELUCons(outChans, elu)
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = nn.Dropout3d() if dropout else nn.Identity()
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(outChans // 2, elu)
        self.relu2 = ELUCons(outChans, elu)
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll, logits=False):
        super().__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(2, elu)
        self.nll = nll
        self.logits = logits

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.view(out.numel() // 2, 2)
        if self.logits:
            return out
        return F.log_softmax(out, dim=1) if self.nll else F.softmax(out, dim=1)


class VNet(nn.Module):
    """V-Net extracted from mattmacy/vnet.pytorch."""

    def __init__(self, elu=True, nll=False, logits=False):
        super().__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll, logits=logits)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
