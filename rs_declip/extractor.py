"""Pretrained Oriented R-CNN backbone/FPN adapter; the detector head is unused."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class ConvModuleNoNorm(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)

class MMDetStyleFPN(nn.Module):

    def __init__(self, in_channels=(256, 512, 1024, 2048), out_channels=256, num_outs=5):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvModuleNoNorm(c, out_channels, 1, 0) for c in in_channels])
        self.fpn_convs = nn.ModuleList([ConvModuleNoNorm(out_channels, out_channels, 3, 1) for _ in in_channels])
        self.num_outs = int(num_outs)

    def forward(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        while len(outs) < self.num_outs:
            outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
        return outs[:self.num_outs]

class OrientedRCNNR50FPNFeatureExtractor(nn.Module):

    def __init__(self, ckpt_path: str, train_backbone: bool=False, train_fpn: bool=False):
        super().__init__()
        rn = resnet50(weights=None)
        self.backbone = nn.Module()
        self.backbone.conv1 = rn.conv1
        self.backbone.bn1 = rn.bn1
        self.backbone.relu = rn.relu
        self.backbone.maxpool = rn.maxpool
        self.backbone.layer1 = rn.layer1
        self.backbone.layer2 = rn.layer2
        self.backbone.layer3 = rn.layer3
        self.backbone.layer4 = rn.layer4
        self.fpn = MMDetStyleFPN(in_channels=(256, 512, 1024, 2048), out_channels=256, num_outs=5)
        self._load_oriented_rcnn_weights(ckpt_path)
        for p in self.backbone.parameters():
            p.requires_grad = bool(train_backbone)
        for p in self.fpn.parameters():
            p.requires_grad = bool(train_fpn)
        self.train_backbone = bool(train_backbone)
        self.train_fpn = bool(train_fpn)

    def _load_oriented_rcnn_weights(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        backbone = {key[len('backbone.'):]: value for key, value in state.items() if key.startswith('backbone.')}
        neck = {key[len('neck.'):]: value for key, value in state.items() if key.startswith('neck.')}
        if not backbone or not neck:
            raise ValueError('A detector checkpoint containing both backbone.* and neck.* weights is required.')
        self.backbone.load_state_dict(backbone, strict=True)
        self.fpn.load_state_dict(neck, strict=True)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        if not self.train_fpn:
            self.fpn.eval()
        return self

    def forward(self, x: torch.Tensor):
        c1 = self.backbone.conv1(x)
        c1 = self.backbone.bn1(c1)
        c1 = self.backbone.relu(c1)
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return self.fpn([c2, c3, c4, c5])
