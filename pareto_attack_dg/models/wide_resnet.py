from __future__ import annotations

import torch
import torch.nn as nn


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = (
            nn.Identity()
            if self.equal_in_out
            else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.equal_in_out:
            out = self.relu1(self.bn1(x))
            shortcut = x
        else:
            out = self.relu1(self.bn1(x))
            shortcut = self.shortcut(out)

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return shortcut + out


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 10,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WideResNet depth must satisfy depth = 6n + 4, got {depth}.")

        blocks_per_group = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = self._make_group(widths[0], widths[1], blocks_per_group, stride=1, dropout_rate=dropout_rate)
        self.block2 = self._make_group(widths[1], widths[2], blocks_per_group, stride=2, dropout_rate=dropout_rate)
        self.block3 = self._make_group(widths[2], widths[3], blocks_per_group, stride=2, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)

        self._initialize()

    def _make_group(
        self,
        in_planes: int,
        out_planes: int,
        blocks: int,
        stride: int,
        dropout_rate: float,
    ) -> nn.Sequential:
        layers = [WideBasicBlock(in_planes, out_planes, stride=stride, dropout_rate=dropout_rate)]
        for _ in range(1, blocks):
            layers.append(WideBasicBlock(out_planes, out_planes, stride=1, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def wrn_28_10_cifar(num_classes: int = 10, dropout_rate: float = 0.0) -> nn.Module:
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropout_rate=dropout_rate)
