# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Downsample with sparse CNN

import os
import sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2'))
from autoencoder import InceptionResNet, make_layer


def make_sparse_down_block(in_dim, hidden_dim, out_dim, doLastRelu=False):
    """
    Make a down-sampling block based on IRN 
    """
    layers = [
        ME.MinkowskiConvolution(in_channels=in_dim, out_channels=hidden_dim,
            kernel_size=3, stride=1, bias=True, dimension=3),
        ME.MinkowskiReLU(inplace=True),
        ME.MinkowskiConvolution(in_channels=hidden_dim, out_channels=out_dim,
            kernel_size=2, stride=2, bias=True, dimension=3),
        ME.MinkowskiReLU(inplace=True),
        make_layer(block=InceptionResNet, block_layers=3, channels=out_dim),
    ]
    if doLastRelu: layers.append(ME.MinkowskiReLU(inplace=True))
    return torch.nn.Sequential(*layers)


class SparseCnnDown1(nn.Module):
    """
    SparseCnnDown module: Down-sample for one time
    """

    def __init__(self, net_config, **kwargs):
        super(SparseCnnDown1, self).__init__()

        self.dims = net_config['dims']
        self.down_block0 = make_sparse_down_block(self.dims[0], self.dims[0], self.dims[1], True)
        self.conv_last = ME.MinkowskiConvolution(in_channels=self.dims[1], out_channels=self.dims[2],
            kernel_size=3, stride=1, bias=True, dimension=3)

    def forward(self, x):

        out0 = self.down_block0(x)
        out0 = self.conv_last(out0)
        return out0


class SparseCnnDown2(nn.Module):
    """
    SparseCnnDown module: Down-sample for two times
    """

    def __init__(self, net_config, **kwargs):
        super(SparseCnnDown2, self).__init__()

        self.dims = net_config['dims']
        self.down_block2 = make_sparse_down_block(self.dims[0], self.dims[0], self.dims[1], True)
        self.down_block1 = make_sparse_down_block(self.dims[1], self.dims[1], self.dims[2], False)

    def forward(self, x):

        out2 = self.down_block2(x)
        out1 = self.down_block1(out2)
        return out1
