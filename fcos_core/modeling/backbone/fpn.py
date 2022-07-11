# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])#C5
        C4_inner =  getattr(self, self.inner_blocks[-2])(x[-2])
        #C4_inner =  getattr(self, self.inner_blocks[:-1][::-1])(x[:-1][::-1])#C4
        #C3_inner =  getattr(self, self.inner_blocks[:-2][::-2])(x[:-2][::-2])#C3
        C3_inner =  getattr(self, self.inner_blocks[-3])(x[-3])
        c5s = F.interpolate(
                last_inner, size=(int(C4_inner.shape[-2]), int(C4_inner.shape[-1])),
                mode='nearest'
            )#C5上采样一次到C4
        c5ss = F.interpolate(
                c5s, size=(int(C3_inner.shape[-2]), int(C3_inner.shape[-1])),
                mode='nearest'
            )#C5上采样两次到C3
        c4s = F.interpolate(
                C4_inner, size=(int(C3_inner.shape[-2]), int(C3_inner.shape[-1])),
                mode='nearest'
            )#C4上采样一次到C3
        P3 = c5ss + c4s + C3_inner
        results = []
        results.append(getattr(self, self.layer_blocks[-3])(P3))#p3
        P3D = F.interpolate(
                P3, size=(int(C4_inner.shape[-2]), int(C4_inner.shape[-1])),
                mode='nearest'
            )#P3下采样一次到C4
        P4 = P3D + C4_inner
        results.append(getattr(self, self.layer_blocks[-2])(P4))#p4
        P4D = F.interpolate(
                P3, size=(int(last_inner.shape[-2]), int(last_inner.shape[-1])),
                mode='nearest'
            )#P4下采样一次到C5
        P5 = P4D + last_inner
        results.append(getattr(self, self.layer_blocks[-1])(P5))#p5
        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
