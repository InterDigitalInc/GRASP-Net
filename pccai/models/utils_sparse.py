# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Utility functions for sparse tensors

import torch
import numpy as np
import MinkowskiEngine as ME


def scale_sparse_tensor_batch(x, factor):
    coords = torch.hstack((x.C[:,0:1], (x.C[:,1:]*factor).round().int()))
    feats = torch.ones((len(coords),1)).float()
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=x.device)
    return x


def sort_sparse_tensor_with_dir(sparse_tensor, dir=1):
    """
    Sort points in sparse tensor according to their coordinates.
    """
    vec = sum([sparse_tensor.C.long().cpu()[:, i] * 
        (sparse_tensor.C.cpu().max().long() + 1) ** (i if dir==0 else (3 - i)) 
        for i in range(sparse_tensor.C.shape[-1])])
    indices_sort = np.argsort(vec)
    sparse_tensor_sort = ME.SparseTensor(features=sparse_tensor.F[indices_sort], 
                                         coordinates=sparse_tensor.C[indices_sort],
                                         tensor_stride=sparse_tensor.tensor_stride[0], 
                                         device=sparse_tensor.device)
    return sparse_tensor_sort


def slice_sparse_tensor(x, slice):
    '''
    A simple function to slice a sparse tensor, can at most get 8 slices
    '''

    if slice == 0: return [x]
    vars = torch.var(x.C[:, 1:].cpu().float(), dim=0).numpy()
    thres = np.percentile(x.C[:, 1:].cpu().numpy(), 50, axis=0)
    axis_l = [AxisSlice('x', vars[0], thres[0], x.C[:, 1] < thres[0]),
                 AxisSlice('x', vars[1], thres[1], x.C[:,2] < thres[1]),
                 AxisSlice('x', vars[2], thres[2], x.C[:,3] < thres[2])]
    axis_l = sorted(axis_l, key=lambda axis: axis.var, reverse=True)

    x_list = []
    if slice == 1:
        masks = [
            axis_l[0].mask,
            axis_l[0].nm(),
        ]
    elif slice == 2:
        masks = [
            torch.logical_and(axis_l[0].mask, axis_l[1].mask),
            torch.logical_and(axis_l[0].nm(), axis_l[1].mask),
            torch.logical_and(axis_l[0].mask, axis_l[1].nm()),
            torch.logical_and(axis_l[0].nm(), axis_l[1].nm())
        ]
    elif slice == 3:
        masks = [
            torch.logical_and(torch.logical_and(axis_l[0].mask, axis_l[1].mask), axis_l[2].mask),
            torch.logical_and(torch.logical_and(axis_l[0].nm(), axis_l[1].mask), axis_l[2].mask),
            torch.logical_and(torch.logical_and(axis_l[0].mask, axis_l[1].nm()), axis_l[2].mask),
            torch.logical_and(torch.logical_and(axis_l[0].nm(), axis_l[1].nm()), axis_l[2].mask),
            torch.logical_and(torch.logical_and(axis_l[0].mask, axis_l[1].mask), axis_l[2].nm()),
            torch.logical_and(torch.logical_and(axis_l[0].nm(), axis_l[1].mask), axis_l[2].nm()),
            torch.logical_and(torch.logical_and(axis_l[0].mask, axis_l[1].nm()), axis_l[2].nm()),
            torch.logical_and(torch.logical_and(axis_l[0].nm(), axis_l[1].nm()), axis_l[2].nm())
        ]

    for mask in masks:
        x_list.append(ME.SparseTensor(
                        features=torch.ones((torch.sum(mask), 1)).float(), 
                        coordinates=x.C[mask], 
                        tensor_stride=1, device=x.device))
    return x_list


class AxisSlice:
    def __init__(self, name, var, thres, mask):
        self.name = name
        self.var = var
        self.thres = thres
        self.mask = mask

    def __repr__(self):
        return repr((self.name, self.var, self.thres, self.mask))

    def nm(self):
        return torch.logical_not(self.mask)