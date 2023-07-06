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
# from correlation_package.modules.corr import Correlation1d # from PWC-Net
from networks.submodules import *
import copy
from ofa.elastic_nn.modules.dynamic_layers import DynamicConvLayer, DynamicResBlock, ResidualBlock, DynamicMBConvLayer
from ofa.elastic_nn.modules.dynamic_op import DynamicConv2d as ofa_conv2d, DynamicBatchNorm2d
from ofa.utils.layers import IdentityLayer

from utils.common import val2list, make_divisible, build_activation
from collections import OrderedDict

MAX_RANGE = 400


class DispNetC(nn.Module):

    def __init__(self, resBlock=True, maxdisp=192, input_channel=3, encoder_ratio=16, decoder_ratio=16):
        super(DispNetC, self).__init__()

        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.relu = nn.ReLU(inplace=False)
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio

        self.disp_width = maxdisp // 8 + 16

        # First Block (Extract)
        self.extract_network = ExtractNet(resBlock=resBlock, maxdisp=self.maxdisp, input_channel=input_channel,
                                          encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)

        # Second Block (CUNet)
        self.cunet = CUNet(resBlock=resBlock, maxdisp=self.maxdisp, input_channel=input_channel,
                           encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)

    def forward(self, inputs, enabled_tensorrt=False):
        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim=1)
        img_left = imgs[0]
        img_right = imgs[1]

        # extract features
        conv1_l, conv2_l, conv3a_l, conv3a_r = self.extract_network(inputs)

        # build corr
        out_corr = build_corr(conv3a_l, conv3a_r, max_disp=self.maxdisp // 8 + 16)
        # generate first-stage flows
        dispnetc_flows = self.cunet(inputs, conv1_l, conv2_l, conv3a_l, out_corr)

        return dispnetc_flows

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class ExtractNet(nn.Module):

    def __init__(self, resBlock=True, maxdisp=192, input_channel=3, encoder_ratio=16, decoder_ratio=16
                 ):
        super(ExtractNet, self).__init__()

        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.relu = nn.ReLU(inplace=False)
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio

        self.disp_width = maxdisp // 8 + 16
        self.dilation = 1

        # shrink and extract features

        self.conv1 = conv(self.input_channel, self.basicE, 7, 2)

        if resBlock:

            self.conv2 = ResBlock(self.basicE, self.basicE * 2, stride=2)
            self.conv3 = ResBlock(self.basicE * 2, self.basicE * 4, stride=2)

        else:
            self.conv2 = conv(self.basicE, self.basicE * 2, stride=2)
            self.conv3 = conv(self.basicE * 2, self.basicE * 4, stride=2)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, 0.02 / n)
                # m.weight.data.normal_(0, 0.02)
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):

        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim=1)
        img_left = imgs[0]
        img_right = imgs[1]

        conv1_l = self.conv1(img_left)
        conv2_l = self.conv2(conv1_l)
        conv3a_l = self.conv3(conv2_l)

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3a_r = self.conv3(conv2_r)

        return conv1_l, conv2_l, conv3a_l, conv3a_r

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class CUNet(nn.Module):

    def __init__(self, resBlock=True, maxdisp=192, input_channel=3, encoder_ratio=16, decoder_ratio=16):
        super(CUNet, self).__init__()

        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.relu = nn.ReLU(inplace=False)
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio

        self.disp_width = maxdisp // 8 + 16

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        if resBlock:
            self.conv_redir = ResBlock(self.basicE * 4, self.basicE, stride=1)
            self.conv3_1 = DyRes(MAX_RANGE // 8 + 16 + self.basicE, self.basicE * 4)
            self.conv4 = ResBlock(self.basicE * 4, self.basicE * 8, stride=2)
            self.conv4_1 = ResBlock(self.basicE * 8, self.basicE * 8)
            self.conv5 = ResBlock(self.basicE * 8, self.basicE * 16, stride=2)
            self.conv5_1 = ResBlock(self.basicE * 16, self.basicE * 16)
            self.conv6 = ResBlock(self.basicE * 16, self.basicE * 32, stride=2)
            self.conv6_1 = ResBlock(self.basicE * 32, self.basicE * 32)
        else:
            self.conv_redir = conv(self.basicE * 4, self.basicE, stride=1)
            self.conv3_1 = conv(self.disp_width + self.basicE, self.basicE * 4)
            self.conv4 = conv(self.basicE * 4, self.basicE * 8, stride=2)
            self.conv4_1 = conv(self.basicE * 8, self.basicE * 8)
            self.conv5 = conv(self.basicE * 8, self.basicE * 16, stride=2)
            self.conv5_1 = conv(self.basicE * 16, self.basicE * 16)
            self.conv6 = conv(self.basicE * 16, self.basicE * 32, stride=2)
            self.conv6_1 = conv(self.basicE * 32, self.basicE * 32)

        self.pred_flow6 = predict_flow(self.basicE * 32)  # 最小视差图

        # # iconv with resblock
        # self.iconv5 = ResBlock(1025, 512, 1)
        # self.iconv4 = ResBlock(769, 256, 1)
        # self.iconv3 = ResBlock(385, 128, 1)
        # self.iconv2 = ResBlock(193, 64, 1)
        # self.iconv1 = ResBlock(97, 32, 1)
        # self.iconv0 = ResBlock(20, 16, 1)

        # iconv with deconv  橙色块
        self.iconv5 = nn.ConvTranspose2d((self.basicD + self.basicE) * 16 + 1, self.basicD * 16, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d((self.basicD + self.basicE) * 8 + 1, self.basicD * 8, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d((self.basicD + self.basicE) * 4 + 1, self.basicD * 4, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d((self.basicD + self.basicE) * 2 + 1, self.basicD * 2, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d((self.basicD + self.basicE) * 1 + 1, self.basicD, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(self.basicD + self.input_channel + 1, self.basicD, 3, 1, 1)

        # expand and produce disparity
        self.upconv5 = deconv(self.basicE * 32, self.basicD * 16, 4)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = predict_flow(self.basicD * 16)

        self.upconv4 = deconv(self.basicD * 16, self.basicD * 8, 4)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = predict_flow(self.basicD * 8)

        self.upconv3 = deconv(self.basicD * 8, self.basicD * 4, 4)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = predict_flow(self.basicD * 4)

        self.upconv2 = deconv(self.basicD * 4, self.basicD * 2, 4)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = predict_flow(self.basicD * 2)

        self.upconv1 = deconv(self.basicD * 2, self.basicD, 4)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = predict_flow(self.basicD)

        self.upconv0 = deconv(self.basicD, self.basicD, 4)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow0 = predict_flow(self.basicD)
        # if self.maxdisp == -1:
        #    self.pred_flow0 = predict_flow(16)
        # else:
        #    self.disp_expand = ResBlock(16, self.maxdisp)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, 0.02 / n)
                # m.weight.data.normal_(0, 0.02)
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.freeze()

    def forward(self, inputs, conv1_l, conv2_l, conv3a_l, corr_volume, get_features=False):

        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim=1)
        img_left = imgs[0]
        img_right = imgs[1]

        # Correlate corr3a_l and corr3a_r
        # out_corr = self.corr(conv3a_l, conv3a_r)
        # out_corr = build_corr(conv3a_l, conv3a_r, max_disp=self.disp_width)
        out_corr = self.corr_activation(corr_volume)
        out_conv3a_redir = self.conv_redir(conv3a_l)
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

        conv3b = self.conv3_1(in_conv3b)
        conv4a = self.conv4(conv3b)
        conv4b = self.conv4_1(conv4a)
        conv5a = self.conv5(conv4b)
        conv5b = self.conv5_1(conv5a)
        conv6a = self.conv6(conv5b)
        conv6b = self.conv6_1(conv6a)

        pr6 = self.pred_flow6(conv6b)
        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        pr5 = self.pred_flow5(iconv5)
        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)

        pr4 = self.pred_flow4(iconv4)
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3 = self.pred_flow3(iconv3)
        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)

        pr2 = self.pred_flow2(iconv2)
        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)

        pr1 = self.pred_flow1(iconv1)
        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)

        # predict flow
        pr0 = self.pred_flow0(iconv0)
        pr0 = self.relu(pr0)
        # if self.maxdisp == -1:
        #    pr0 = self.pred_flow0(iconv0)
        #    pr0 = self.relu(pr0)
        # else:
        #    pr0 = self.disp_expand(iconv0)
        #    pr0 = F.softmax(pr0, dim=1)
        #    pr0 = disparity_regression(pr0, self.maxdisp)

        # predict flow from dropout output
        # pr6 = self.pred_flow6(F.dropout2d(conv6b))
        # pr5 = self.pred_flow5(F.dropout2d(iconv5))
        # pr4 = self.pred_flow4(F.dropout2d(iconv4))
        # pr3 = self.pred_flow3(F.dropout2d(iconv3))
        # pr2 = self.pred_flow2(F.dropout2d(iconv2))
        # pr1 = self.pred_flow1(F.dropout2d(iconv1))
        # pr0 = self.pred_flow0(F.dropout2d(iconv0))

        # if self.training:
        #     # print("finish forwarding.")
        #     return pr0, pr1, pr2, pr3, pr4, pr5, pr6
        # else:
        #     return pr0

        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)

        # can be chosen outside
        if get_features:
            features = (iconv5, iconv4, iconv3, iconv2, iconv1, iconv0)
            return disps, features
        else:
            return disps

    def freeze(self):
        for name, param in self.named_parameters():
            if ('weight' in name) or ('bias' in name):
                param.requires_grad = False

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class NAS_CUNet(nn.Module):

    def __init__(self, maxdisp=192, input_channel=3, encoder_ratio=16, decoder_ratio=16,
                 ks_list=7, expand_ratio_list=8, depth_list=4
                 ):
        super(NAS_CUNet, self).__init__()

        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.relu = nn.ReLU(inplace=False)
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio

        self.disp_width = maxdisp // 8 + 16

        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        # inchannel=self.basicE * 4
        self.conv_redir = ResBlock(64, self.basicE, stride=1)
        self.conv3_1 = DyRes(MAX_RANGE // 8 + 16 + self.basicE, self.basicE * 4)

        # self.conv4 = ResBlock(self.basicE * 4, self.basicE * 8, stride=2)
        # self.conv4_1 = ResBlock(self.basicE * 8, self.basicE * 8)
        self.feature_blocks = nn.ModuleList(self.ConstructFeatureNet(self.basicE))

        self.feature_runtime_depth = [len(block_idx) for block_idx in self.fea_block_group_info]
        # self.conv5 = ResBlock(self.basicE * 8, self.basicE * 16, stride=2)
        # self.conv5_1 = ResBlock(self.basicE * 16, self.basicE * 16)

        self.conv6 = ResBlock(self.basicE * 16, self.basicE * 32, stride=2)
        self.conv6_1 = ResBlock(self.basicE * 32, self.basicE * 32)

        self.pred_flow6 = predict_flow(self.basicE * 32)  # 最小视差图

        # iconv with deconv  蓝色块
        self.iconv5 = nn.ConvTranspose2d((self.basicD + self.basicE) * 16 + 1, self.basicD * 16, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d((self.basicD + self.basicE) * 8 + 1, self.basicD * 8, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d((self.basicD + self.basicE) * 4 + 1, self.basicD * 4, 3, 1, 1)
        # self.iconv2 = nn.ConvTranspose2d((self.basicD + self.basicE) * 2 + 1, self.basicD * 2, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(self.basicD * 2 + 32 + 1, self.basicD * 2, 3, 1, 1)  # 2=conv2l.channel
        self.iconv1 = nn.ConvTranspose2d(self.basicD + 24 + 1, self.basicD, 3, 1, 1)  # 64=conv1l.channel
        # self.iconv1 = nn.ConvTranspose2d(self.basicD + self.basicE * 1 + 1, self.basicD, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(self.basicD + self.input_channel + 1, self.basicD, 3, 1, 1)

        # expand and produce disparity

        self.upconv5 = deconv(self.basicE * 32, self.basicD * 16, kernal_size=4)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = predict_flow(self.basicD * 16)

        self.upconv4 = deconv(self.basicD * 16, self.basicD * 8, 4)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = predict_flow(self.basicD * 8)

        self.upconv3 = deconv(self.basicD * 8, self.basicD * 4, 4)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = predict_flow(self.basicD * 4)

        self.upconv2 = deconv(self.basicD * 4, self.basicD * 2, 4)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = predict_flow(self.basicD * 2)

        self.upconv1 = deconv(self.basicD * 2, self.basicD, 4)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = predict_flow(self.basicD)

        self.upconv0 = deconv(self.basicD, self.basicD, 4)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow0 = predict_flow(self.basicD)
        # if self.maxdisp == -1:
        #    self.pred_flow0 = predict_flow(16)
        # else:
        #    self.disp_expand = ResBlock(16, self.maxdisp)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, 0.02 / n)
                # m.weight.data.normal_(0, 0.02)
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.freeze
        self.sample_active_subnet()
        # self.set_active_subnet(ks=self.ks_list, expand_ratio_list=self.expand_ratio_list, depth_list=self.depth_list)

    def forward(self, inputs, conv1_l, conv2_l, conv3a_l, corr_volume, get_features=False):

        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim=1)
        img_left = imgs[0]
        img_right = imgs[1]

        # Correlate corr3a_l and corr3a_r
        # out_corr = self.corr(conv3a_l, conv3a_r)
        # out_corr = build_corr(conv3a_l, conv3a_r, max_disp=self.disp_width)
        out_corr = self.corr_activation(corr_volume)
        out_conv3a_redir = self.conv_redir(conv3a_l)
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

        conv3b = self.conv3_1(in_conv3b)

        conv4, conv5 = self.feature_extraction(conv3b)

        conv6a = self.conv6(conv5)
        conv6b = self.conv6_1(conv6a)

        pr6 = self.pred_flow6(conv6b)
        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5), 1)
        iconv5 = self.iconv5(concat5)  # [4,512,12,24]

        pr5 = self.pred_flow5(iconv5)  # [4,512,12,24]
        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4), 1)
        iconv4 = self.iconv4(concat4)

        pr4 = self.pred_flow4(iconv4)
        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3 = self.pred_flow3(iconv3)
        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)

        pr2 = self.pred_flow2(iconv2)
        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)

        pr1 = self.pred_flow1(iconv1)
        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)

        # predict flow
        pr0 = self.pred_flow0(iconv0)
        pr0 = self.relu(pr0)
        # if self.maxdisp == -1:
        #    pr0 = self.pred_flow0(iconv0)
        #    pr0 = self.relu(pr0)
        # else:
        #    pr0 = self.disp_expand(iconv0)
        #    pr0 = F.softmax(pr0, dim=1)
        #    pr0 = disparity_regression(pr0, self.maxdisp)

        # predict flow from dropout output
        # pr6 = self.pred_flow6(F.dropout2d(conv6b))
        # pr5 = self.pred_flow5(F.dropout2d(iconv5))
        # pr4 = self.pred_flow4(F.dropout2d(iconv4))
        # pr3 = self.pred_flow3(F.dropout2d(iconv3))
        # pr2 = self.pred_flow2(F.dropout2d(iconv2))
        # pr1 = self.pred_flow1(F.dropout2d(iconv1))
        # pr0 = self.pred_flow0(F.dropout2d(iconv0))

        # if self.training:
        #     # print("finish forwarding.")
        #     return pr0, pr1, pr2, pr3, pr4, pr5, pr6
        # else:
        #     return pr0

        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)

        # can be chosen outside
        if get_features:
            features = (iconv5, iconv4, iconv3, iconv2, iconv1, iconv0)
            return disps, features
        else:
            return disps

    def freeze(self):
        for name, param in self.named_parameters():
            if ('weight' in name) or ('bias' in name):
                param.requires_grad = False

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def ConstructFeatureNet(self, input_channel):
        width_list = [input_channel * 8, input_channel * 16]
        n_feature_blocks = [max(self.depth_list)] * 2
        stride_stages = [2, 2]
        act_stages = ['relu', 'relu']
        se_stages = [False, True]
        self.fea_block_group_info = []
        blocks = []
        _block_index = 0

        feature_dim = self.basicE * 4
        for width, n_block, s, act_func, use_se in zip(width_list, n_feature_blocks, stride_stages, act_stages,
                                                       se_stages):
            self.fea_block_group_info.append([_block_index + i for i in range(n_block)])
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
        # conv4 = blocks[:4]
        # conv5 = blocks[4:]
        return blocks

    def feature_extraction(self, input):
        out = input
        conv4conv5 = []
        for stage_id, block_idx in enumerate(self.fea_block_group_info):
            depth = self.feature_runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                out = self.feature_blocks[idx](out)
            conv4conv5.append(out)
        conv4 = conv4conv5[0]
        conv5 = conv4conv5[1]
        return conv4, conv5

    @property
    def config(self):
        return {
            'name': NAS_CUNet.__name__,
            'bn': self.get_bn_param(),
            'fea_blocks': [
                block.config for block in self.feature_blocks
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
        super(NAS_CUNet, self).load_state_dict(model_dict)

    def set_max_net(self):
        self.set_active_subnet(ks=max(self.ks_list), es=max(self.expand_ratio_list), ds=max(self.depth_list),
                               )

    def set_active_subnet(self, ks=None, es=None, ds=None, **kwargs):
        # featureblock
        cunet_ks = val2list(ks, len(self.feature_blocks))
        cunet_expand_ratio = val2list(es, len(self.feature_blocks))
        cunet_depth = val2list(ds, len(self.fea_block_group_info))

        for block, k, e in zip(self.feature_blocks, cunet_ks, cunet_expand_ratio):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(cunet_depth):
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
            depth_candidates = [depth_candidates for _ in range(len(self.fea_block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            'cunet_ks': ks_setting,
            'cunet_es': expand_setting,
            'cunet_ds': depth_setting,
        }

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




    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.feature_blocks:
            block.conv.re_organize_middle_weights(expand_ratio_stage)