from __future__ import print_function

import random
from os import system

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal
from torchinfo import summary

# from torch2trt import torch2trt
from networks.FADNet import FADNet
from networks.DispNetC import ExtractNet, CUNet
from networks.DispNetRes import DispNetRes
from networks.submodules import *
from utils.common import val2list, make_divisible
from ofa.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer
from ofa.utils.layers import IdentityLayer, ResidualBlock
from ofa.utils.my_modules import MyNetwork, get_bn_param
import copy


# from torch2trt import torch2trt

class OFAFADNet(FADNet):

    def __init__(self, width_mult=1.0,
                 ks_list=7, expand_ratio_list=8, depth_list=4, scale_list=3, encoder_ratio=16,
                 decoder_ratio=16):

        # Add by jingbo
        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.scale_list = val2list(scale_list, 1)
        self.active_scale = max(self.scale_list)
        self.max_scale = max(self.scale_list)
        self.active_scale = self.max_scale

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()
        self.scale_list.sort()

        # for fad_cuent_nas
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio

        self.feature_blocks = self.ConstructFeatureNet()  # 12 blocks

        self.feature_runtime_depth = [len(block_idx) for block_idx in self.fea_block_group_info]  # [4,4,4]

        super(OFAFADNet, self).__init__(maxdisp=192, feature_blocks=self.feature_blocks,
                                        num_scales=max(self.scale_list), ks_list=self.ks_list,
                                        expand_ratio_list=self.expand_ratio_list, depth_list=self.depth_list
                                        )

    def ConstructFeatureNet(self):
        base_stage_width = [24] + [16 * (2 ** i) for i in range(1, max(self.scale_list))]

        stride_stages = [2] + [2 for _ in range(max(self.scale_list) - 1)]
        act_stages = ['relu' for _ in range(max(self.scale_list))]  # relu

        se_stages = [False if i % 2 == 0 else True for i in range(self.max_scale)]
        n_block_list = [max(self.depth_list)] * max(self.scale_list)  # [4]* 3
        width_list = []  # [24]
        for base_width in base_stage_width:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)

        # inverted residual blocks
        self.fea_block_group_info = []
        blocks = []
        _block_index = 0

        feature_dim = 3
        for width, n_block, s, act_func, use_se in zip(width_list, n_block_list,
                                                       stride_stages, act_stages, se_stages):
            self.fea_block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width  # 24
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                    kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list,
                    stride=stride, act_func=act_func, use_se=use_se,
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

        return blocks

    def Construct_cunet_nas_Feature(self, input_channel):
        width_list = [input_channel * 8, input_channel * 16]
        # n_feature_blocks = [max(self.depth_list)] * 2
        n_feature_blocks = [4, 4]
        stride_stages = [2, 2]
        act_stages = ['relu', 'relu']
        se_stages = [False, True]
        # n_block_list = [max(self.depth_list)] * max(self.scale_list)

        self.cunet_nas_group_info = []
        blocks = []
        _block_index = 0

        feature_dim = self.basicE * 4
        for width, n_block, s, act_func, use_se in zip(width_list, n_feature_blocks, stride_stages, act_stages,
                                                       se_stages):
            self.cunet_nas_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
                    kernel_size_list=self.ks_list, expand_ratio_list=self.expand_ratio_list, use_se=use_se,
                    stride=stride, act_func=act_func
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = IdentityLayer(feature_dim, feature_dim)
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel

        return blocks

    def feature_extraction(self, inputs):
        imgs = torch.chunk(inputs, 2, dim=1)
        img_left = imgs[0]
        img_right = imgs[1]
        outl = img_left
        outr = img_right

        left_temp = []
        right_temp = []
        for stage_id, block_idx in enumerate(self.fea_block_group_info):
            # if stage_id < self.active_scale:  # 根据active_scale选取前几个特征阶段
            depth = self.feature_runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                outl = self.feature_blocks[idx](outl)
            left_temp.append(outl)
            for idx in active_idx:
                outr = self.feature_blocks[idx](outr)
            right_temp.append(outr)
        conv1_l = left_temp[0]
        conv2_l = left_temp[1]
        conv3a_l = left_temp[2]
        conv3a_r = right_temp[2]
        self.conv2_l_channel = conv2_l.size()[1]
        return conv1_l, conv2_l, conv3a_l, conv3a_r

    @property
    def config(self):
        return {
            'name': OFAFADNet.__name__,
            'bn': self.get_bn_param(),
            'fea_blocks': [
                block.config for block in self.feature_blocks
            ],
            'cunet_nas_blocks': [
                block.config for block in self.cunet_nas_blocks
            ]
        }

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if '.mobile_inverted_conv.' in key:
                new_key = key.replace('.mobile_inverted_conv.', '.conv.')
            else:
                new_key = key
            if new_key in model_dict:
                pass
            elif '.bn.bn.' in new_key:
                new_key = new_key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in new_key:
                new_key = new_key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in new_key:
                new_key = new_key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in new_key:
                new_key = new_key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in new_key:
                new_key = new_key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in new_key:
                new_key = new_key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAFADNet, self).load_state_dict(model_dict)

    def set_max_net(self):
        self.set_active_subnet(ks=max(self.ks_list), es=max(self.expand_ratio_list), ds=max(self.depth_list)
                               )
        self.cunet.set_active_subnet(ks=max(self.ks_list), es=max(self.expand_ratio_list), ds=max(self.depth_list))

    def set_active_subnet(self, ks=None, es=None, ds=None, **kwargs):

        fad_ks = val2list(ks, len(self.feature_blocks))
        fad_expand_ratio = val2list(es, len(self.feature_blocks))
        fad_depth = val2list(ds, len(self.fea_block_group_info))

        for block, k, e in zip(self.feature_blocks[:], fad_ks, fad_expand_ratio):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(fad_depth):
            if d is not None:
                self.feature_runtime_depth[i] = min(len(self.fea_block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'depth':
            self.__dict__['_depth_include_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        elif constraint_type == 'scale':
            self.__dict__['_scale_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_ks_include_list'] = None
        self.__dict__['_scale_include_list'] = None

    def sample_active_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']
        scale_candidates = self.scale_list if self.__dict__.get('_scale_include_list', None) is None else \
            self.__dict__['_scale_include_list']

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.feature_blocks))]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.feature_blocks))]

        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):  # depth_candidates=[4]
            depth_candidates = [depth_candidates for _ in
                                range(len(self.fea_block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        cunet_settings = self.cunet.sample_active_subnet()
        self.set_active_subnet(ks_setting, expand_setting, depth_setting)
        ofa_settings = {
            'ks': ks_setting,
            'es': expand_setting,
            'ds': depth_setting
        }
        ofa_settings.update(cunet_settings)
        return ofa_settings

    def get_active_subnet(self, preserve_weight=True):

        blocks = []
        cunet_block = []
        input_channel = 3
        # feature blocks
        for stage_id, block_idx in enumerate(self.fea_block_group_info):
            depth = self.feature_runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(ResidualBlock(
                    self.feature_blocks[idx].conv.get_active_subnet(input_channel, preserve_weight),
                    copy.deepcopy(self.feature_blocks[idx].shortcut)
                ))
                input_channel = stage_blocks[-1].conv.out_channels
            blocks += stage_blocks

        _subnet = FADNet(maxdisp=192, feature_blocks=blocks,
                         num_scales=self.active_scale)

        # make deep copy of other modules
        _subnet.cunet = copy.deepcopy(self.cunet)
        _subnet.dispnetres = copy.deepcopy(self.dispnetres)

        return _subnet

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.feature_blocks:
            block.conv.re_organize_middle_weights(expand_ratio_stage)

        self.cunet.re_organize_middle_weights(expand_ratio_stage=expand_ratio_stage)

    @property
    def grouped_block_index(self):
        return self.fea_block_group_info

    def get_bn_param(self):
        return get_bn_param(self)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


if __name__ == '__main__':
    model = OFAFADNet()

    # print(model.cunet)
    # print(model.cunet.feature_runtime_depth)
    # print(model)
