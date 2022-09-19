# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# MLP Decoder implemented with MinkowskiEngine operated on sparse tensors 

import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME

def make_pointwise_mlp_sparse(dims, doLastRelu=False):
    """
    Make poinwise MLP layers based on MinkowskiEngine
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(
            ME.MinkowskiLinear(dims[i], dims[i + 1], bias=True)
        )
        if i != len(dims) - 2 or doLastRelu:
            layers.append(ME.MinkowskiReLU(inplace=True))
    return torch.nn.Sequential(*layers)

class MlpDecoderSparse(nn.Module):
    """
    MLP decoder implemented with MinkowskiEngine
    """

    def __init__(self, net_config, **kwargs):
        super(MlpDecoderSparse, self).__init__()
        self.num_points = net_config['num_points']
        dims = net_config['dims']
        self.mlp = make_pointwise_mlp_sparse(dims + [3 * self.num_points], doLastRelu=False) # the MLP layers

    def forward(self, x):
        out = self.mlp(x) # BatchSize X PointNum X 3
        return out
