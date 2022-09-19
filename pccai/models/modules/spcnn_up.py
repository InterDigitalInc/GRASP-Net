# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Upsample with sparse CNN

import os
import sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2'))
from data_utils import isin
from autoencoder import InceptionResNet, make_layer


def make_sparse_up_block(in_dim, hidden_dim, out_dim, doLastRelu):
    """
    Make a up-sampling block based on IRN 
    """
    layers = [
        ME.MinkowskiGenerativeConvolutionTranspose(in_channels=in_dim, out_channels=hidden_dim,
            kernel_size=2, stride=2, bias=True, dimension=3),
        ME.MinkowskiReLU(inplace=True),
        ME.MinkowskiConvolution(in_channels=hidden_dim, out_channels=out_dim,
            kernel_size=3, stride=1, bias=True, dimension=3),
        ME.MinkowskiReLU(inplace=True),
        make_layer(block=InceptionResNet, block_layers=3, channels=out_dim),
    ]
    if doLastRelu: layers.append(ME.MinkowskiReLU(inplace=True))
    return torch.nn.Sequential(*layers)


class SparseCnnUp1(nn.Module):
    """
    SparseCnnUp module: Up-sample for one time
    """

    def __init__(self, net_config, **kwargs):
        super(SparseCnnUp1, self).__init__()

        self.dims = net_config['dims']
        self.up_block0 = make_sparse_up_block(self.dims[0], self.dims[1], self.dims[1], False)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, y1, gt_pc): # from coarse to fine

        out = self.up_block0(y1)
        out = self.prune_voxel(out, gt_pc.C)
        return out

    def prune_voxel(self, coarse_voxels, refined_voxels):
        mask = isin(coarse_voxels.C, refined_voxels)
        data_pruned = self.pruning(coarse_voxels, mask.to(coarse_voxels.device))
        return data_pruned


class SparseCnnUp2(nn.Module):
    """
    SparseCnnUp2 module: Up-sample for two times
    """

    def __init__(self, net_config, **kwargs):
        super(SparseCnnUp2, self).__init__()

        self.dims = net_config['dims']
        self.up_block1 = make_sparse_up_block(self.dims[0], self.dims[1], self.dims[1], True)
        self.up_block2 = make_sparse_up_block(self.dims[1], self.dims[2], self.dims[2], False)

        self.pruning = ME.MinkowskiPruning()
        self.pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)

    def forward(self, y1, gt_pc): # from coarse to fine

        # Upsample for the first time
        out = self.up_block1(y1)
        y2_C = self.pool(gt_pc)
        out = SparseCnnUp1.prune_voxel(self, out, y2_C.C)

        # Upsample for the second time
        out = self.up_block2(out)
        out = SparseCnnUp1.prune_voxel(self, out, gt_pc.C)

        return out
