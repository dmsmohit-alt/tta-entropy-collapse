"""CIFAR-style ResNet-18 (3x3 stem, no max-pool) with input normalisation built in.

Convention used across the project: every model takes images as float tensors in [0, 1] (NCHW) and
normalises internally, so TTA code never has to know the dataset statistics.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Normalize(nn.Module):
    def __init__(self, mean: tuple[float, ...] = CIFAR10_MEAN, std: tuple[float, ...] = CIFAR10_STD) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.normalize = Normalize()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        layers, in_planes = [], 64
        for planes, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            for s in (stride, 1):                           # two BasicBlocks per stage
                layers.append(BasicBlock(in_planes, planes, s))
                in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(self.normalize(x))))
        out = self.layers(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
