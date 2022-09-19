# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


# Codec for the GRASP-Net

import os
import sys
import time
import numpy as np

# Need to put it here due to unknown conflict with MinkowskiEngine
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party/nndistance'))

import torch
import MinkowskiEngine as ME
from pccai.codecs.pcc_codec import PccCodecBase
from pccai.models.utils_sparse import slice_sparse_tensor

try:
    import faiss
    found_FAISS = True
except ModuleNotFoundError:
    found_FAISS = False

class GeoResCompressionCodec(PccCodecBase):
    """
    Geometric Residual Analysis and Synthesis for PCC, m58962, Jan 22, the codec itself
    """

    def __init__(self, codec_config, pccnet, bit_depth, syntax):
        super().__init__(codec_config, pccnet, syntax)
        self.res = 2 ** bit_depth
        pccnet.base_only = codec_config.get('base_only', False) # whether to use FAISS for NN search
        if pccnet.skip_mode == False:
            pccnet.res_enc.faiss = codec_config.get('faiss', True) and found_FAISS == True # overwrite the option of whether to use FAISS for NN search

        # Set the slice parameter
        self.slice = codec_config.get('slice', 0)

        # For surface point clouds (<=16bit), slice parameter is set empirically to avoid high memory cost
        if bit_depth >= 12 and bit_depth <= 16:
            if pccnet.scaling_ratio >= 0.625:
                self.slice = 3
            elif pccnet.scaling_ratio >= 0.5:
                self.slice = 2
            elif pccnet.scaling_ratio >= 0.375:
                self.slice = 1

    def compress(self, coords, tag):
        """
        Compress all the transform blocks in a point cloud and write the bitstream to a file.
        """

        # Construct the sparse tensor from the point cloud, slice it if necessary
        start = time.monotonic()
        coords = (coords + np.array(self.translate)) * self.scale # normalize
        pnt_cnt = coords.shape[0]
        coords = torch.tensor(coords)
        feats = torch.ones((len(coords), 1)).float()
        coords, feats = ME.utils.sparse_collate([coords], [feats])
        device = next(self.pccnet.parameters()).device
        x_list = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)
        x_list = slice_sparse_tensor(x_list, self.slice)
        end = time.monotonic()

        # Preparations
        filename_list = []
        stat_dict = {
            'scaled_num_points': 0,
            'all_enc_time': end - start,
            'base_enc_time': 0,
            'bpp_base': 0
        }
        
        for cnt_slice, x in enumerate(x_list):

            # Compress with the network
            start = time.monotonic()
            filename_base, string_set, min_v_set, max_v_set, shape_set, scaled_num_points, base_enc_time = self.pccnet.compress(x, tag + '_' + str(cnt_slice))
            end = time.monotonic()
            stat_dict['scaled_num_points'] += scaled_num_points
            stat_dict['all_enc_time'] += end - start
            stat_dict['base_enc_time'] += base_enc_time
            stat_dict['bpp_base'] += os.stat(filename_base).st_size * 8 / pnt_cnt
            filename_list.append(filename_base)
            if self.pccnet.skip_mode == False:
                filename_enhance = tag + '_' + str(cnt_slice) + '_E' + '.bin'
                filename_header = tag + '_' + str(cnt_slice) + '_H' + '.bin'

                # Write down the strings
                with open(filename_enhance, 'wb') as fout:
                    for cnt, string in enumerate(string_set):
                        fout.write(string)
                        key = 'bpp_feat'
                        if cnt >= 1: key += '_res' + str(len(string_set) - cnt - 1)
                        if (key in stat_dict) == False: stat_dict[key] = 0
                        stat_dict[key] += len(string) * 8 / pnt_cnt

                # Write down the headers
                with open(filename_header, 'wb') as fout:
                    for cnt in range(len(string_set)):
                        fout.write(np.array(shape_set[cnt], dtype=np.int32).tobytes())
                        fout.write(np.array(len(min_v_set[cnt]), dtype=np.int8).tobytes())
                        fout.write(np.array(min_v_set[cnt], dtype=np.float32).tobytes())
                        fout.write(np.array(max_v_set[cnt], dtype=np.float32).tobytes())
                        if cnt != len(string_set) - 1:
                            fout.write(np.array(len(string_set[cnt]), dtype=np.int32).tobytes())

                filename_list.append(filename_enhance)
                filename_list.append(filename_header)

        # Convert the extra statistics to string for logging
        stat_dict['scaled_num_points'] = stat_dict['scaled_num_points']
        stat_dict['enc_time'] = round(stat_dict['all_enc_time'] - stat_dict['base_enc_time'], 3)
        stat_dict['all_enc_time'] = round(stat_dict['all_enc_time'], 3)
        stat_dict['base_enc_time'] = round(stat_dict['base_enc_time'], 3)
        for k, v in stat_dict.items():
            if k.find('bpp_') == 0:
                stat_dict[k] = round(v, 6)
        return filename_list, stat_dict


    def decompress(self, filename):
        """
        Decompress all the transform blocks of a point cloud from a file.
        """
        stat_dict = {
            'all_dec_time': 0,
            'base_dec_time': 0,
        }
        for cnt_slice in range(2 ** self.slice):

            # Read the strings
            if self.pccnet.skip_mode == False:
                with open(filename[cnt_slice * 3 + 1], 'rb') as fin:
                    string_set = [fin.read()]
                shape_set, min_v_set, max_v_set = [], [], []

                # Read the header, then parse the strings accordingly
                with open(filename[cnt_slice * 3 + 2], 'rb') as fin:
                    shape_set.append(np.frombuffer(fin.read(4*2), dtype=np.int32))
                    len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
                    min_v_set.append(np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0])
                    max_v_set.append(np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0])

            else:
                string_set, min_v_set, max_v_set, shape_set = None, None, None, None
            
            # Perform decompression
            base_dec_time = [0]
            start = time.monotonic()
            if cnt_slice == 0:
                pc_rec = self.postprocess(
                    self.pccnet.decompress(filename[0], string_set, min_v_set, max_v_set, shape_set, base_dec_time)
                )
            else:
                pc_rec = torch.vstack((
                    pc_rec, self.postprocess(
                        self.pccnet.decompress(filename[cnt_slice * (1 if self.pccnet.skip_mode else 3)], 
                            string_set, min_v_set, max_v_set, shape_set, base_dec_time))
                    )
                )
            end = time.monotonic()
    
            stat_dict['all_dec_time'] += end - start
            stat_dict['base_dec_time'] += base_dec_time[0]

        stat_dict['dec_time'] = round(stat_dict['all_dec_time'] - stat_dict['base_dec_time'], 3)
        stat_dict['all_dec_time'] = round(stat_dict['all_dec_time'], 3)
        stat_dict['base_dec_time'] = round(stat_dict['base_dec_time'], 3)
        return pc_rec, stat_dict


    def postprocess(self, pc_rec):
        """
        Postprocessing after the point cloud is decompressed
        """
        # Clip the overflow values
        pc_rec = pc_rec.round().long()
        pc_rec[pc_rec[:, 0] >= self.res, 0] = self.res - 1
        pc_rec[pc_rec[:, 1] >= self.res, 1] = self.res - 1
        pc_rec[pc_rec[:, 2] >= self.res, 2] = self.res - 1
        pc_rec[pc_rec[:, 0] < 0, 0] = 0
        pc_rec[pc_rec[:, 1] < 0, 1] = 0
        pc_rec[pc_rec[:, 2] < 0, 2] = 0

        # Keep unique points only
        pc_rec = pc_rec[:,0] * (self.res ** 2) + pc_rec[:, 1] * self.res + pc_rec[:, 2]
        pc_rec = torch.unique(pc_rec)
        out0 = torch.floor(pc_rec / (self.res ** 2)).long()
        pc_rec = pc_rec - out0 * (self.res ** 2)
        out1 = torch.floor(pc_rec / self.res).long()
        pc_rec = pc_rec - out1 * self.res
        pc_rec = torch.cat([out0.unsqueeze(1), out1.unsqueeze(1), pc_rec.unsqueeze(1)], dim=1)
        pc_rec = (pc_rec / self.scale - torch.tensor(self.translate, device=pc_rec.device)).long() # denormalize
        return pc_rec
