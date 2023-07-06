from __future__ import print_function
import os, sys, gc
import time
import random

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from net_builder import build_net
from dataloader.SceneFlowLoader import SceneFlowDataset
from dataloader.SintelLoader import SintelDataset
from dataloader.MiddleburyLoader import MiddleburyDataset
from dataloader.ETH3DLoader import ETH3DDataset
from dataloader.RVCLoader import RVCDataset
from dataloader import KITTILoader as DA
from dataloader.GANet.data import get_training_set, get_test_set
from utils.AverageMeter import AverageMeter, HVDMetric
from utils.common import logger, MultiEpochsDataLoader, count_parameters
from losses.multiscaleloss import EPE, d1_metric
from utils.preprocess import scale_disp
from lamb import Lamb
from ofa.elastic_nn.utils import set_running_statistics
import skimage
from torch.nn.parallel import DistributedDataParallel as DDP


class DisparityTrainer(object):
    def __init__(self, net_name, lr, devices, dataset, trainlist, vallist, datapath, batch_size, maxdisp, pretrain=None,
                 ngpu=1, rank=0, hvd=False):
        super(DisparityTrainer, self).__init__()
        self.net_name = net_name
        self.lr = lr
        self.current_lr = lr
        if isinstance(devices, list):
            self.devices = devices
            self.ngpu = ngpu
        elif isinstance(devices, str):
            self.devices = [int(item) for item in devices.split(',')]
            self.ngpu = len(devices)
        self.rank = rank
        self.hvd = hvd
        self.trainlist = trainlist
        self.vallist = vallist
        self.dataset = dataset
        self.datapath = datapath
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.maxdisp = maxdisp

        self.criterion = None
        self.epe = EPE
        self.train_iter = 0
        self.initialize()

    def _prepare_dataset(self):
        if self.dataset == 'sceneflow' or self.dataset == 'irs':
            train_dataset = SceneFlowDataset(txt_file=self.trainlist, root_dir=self.datapath, phase='train')
            test_dataset = SceneFlowDataset(txt_file=self.vallist, root_dir=self.datapath, phase='test')
        if self.dataset == 'middlebury':
            train_dataset = MiddleburyDataset(txt_file=self.trainlist, root_dir=self.datapath, phase='train')
            test_dataset = MiddleburyDataset(txt_file=self.vallist, root_dir=self.datapath, phase='test')
        if self.dataset == 'eth3d':
            train_dataset = ETH3DDataset(txt_file=self.trainlist, root_dir=self.datapath, phase='train')
            test_dataset = ETH3DDataset(txt_file=self.vallist, root_dir=self.datapath, phase='test')
        if self.dataset == 'rvc':
            train_dataset = RVCDataset(txt_file=self.trainlist, root_dir=self.datapath, phase='train')
            test_dataset = RVCDataset(txt_file=self.vallist, root_dir=self.datapath, phase='test')
        if self.dataset == 'sintel':
            train_dataset = SintelDataset(txt_file=self.trainlist, root_dir=self.datapath, phase='train')
            test_dataset = SintelDataset(txt_file=self.vallist, root_dir=self.datapath, phase='test')
        if self.dataset == 'kitti2012':
            from dataloader import KITTIloader2012 as ls
            all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
                self.datapath)
            train_dataset = DA.myImageFolder(all_left_img, all_right_img, all_left_disp, True)
            test_dataset = DA.myImageFolder(test_left_img, test_right_img, test_left_disp, False)
        if self.dataset == 'kitti2015':
            from dataloader import KITTIloader2015 as ls
            all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
                self.datapath)
            train_dataset = DA.myImageFolder(all_left_img, all_right_img, all_left_disp, True)
            test_dataset = DA.myImageFolder(test_left_img, test_right_img, test_left_disp, False)

        self.img_height, self.img_width = train_dataset.get_img_size()
        self.scale_size = train_dataset.get_scale_size()

        datathread = 2
        if os.environ.get('datathread') is not None:
            datathread = int(os.environ.get('datathread'))
        logger.info("Use %d processes to load data..." % datathread)
        # self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, \
        #                        shuffle = True, num_workers = datathread, \
        #                        pin_memory = True)
        train_sampler = None
        shuffle = True
        if self.ngpu > 1 and self.hvd:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=self.ngpu, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        # self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, \
        self.train_loader = MultiEpochsDataLoader(train_dataset, batch_size=self.batch_size, \
                                                  shuffle=shuffle, num_workers=datathread, \
                                                  pin_memory=True, sampler=train_sampler)

        # self.test_loader = DataLoader(test_dataset, batch_size = self.batch_size, \
        #                        shuffle = False, num_workers = datathread, \
        #                        pin_memory = True)
        if self.ngpu > 1 and self.hvd:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=self.ngpu,
                                                                           rank=self.rank)
            # self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler)
            self.test_loader = MultiEpochsDataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler)
        else:
            self.test_loader = MultiEpochsDataLoader(test_dataset, batch_size=self.batch_size, \
                                                     shuffle=False, num_workers=datathread, \
                                                     pin_memory=True)
        self.num_batches_per_epoch = len(self.train_loader)

    def _build_net(self):

        # build net according to the net name
        if self.net_name in ["ucnet", "ucresnet", "psmnet", "ganet", "gwcnet", "aanet"]:
            self.net = build_net(self.net_name)(self.maxdisp)
        elif self.net_name in ['fadnet', 'dispnetcss', 'mobilefadnet', 'slightfadnet', 'tinyfadnet',
                               'microfadnet',
                               'xfadnet']:
            eratio = 16;
            dratio = 16
            if self.net_name == 'mobilefadnet':
                eratio = 8;
                dratio = 8
            elif self.net_name == 'slightfadnet':
                eratio = 4;
                dratio = 4
            elif self.net_name == 'tinyfadnet':
                eratio = 2;
                dratio = 1
            elif self.net_name == 'microfadnet':
                eratio = 1;
                dratio = 1
            elif self.net_name == 'xfadnet':
                eratio = 32;
                dratio = 32
            self.net = build_net(self.net_name)(maxdisp=self.maxdisp, encoder_ratio=eratio, decoder_ratio=dratio)

        if self.net_name in ['ofa_fadnet']:
            self.net = build_net(self.net_name)(width_mult=1.0,
                                                ks_list=7, expand_ratio_list=8, depth_list=4,
                                                scale_list=3,
                                                encoder_ratio=16, decoder_ratio=16)
        self.is_pretrain = False

        if self.pretrain == '':
            logger.info('Initial a new model...')
        else:
            if os.path.isfile(self.pretrain):
                model_data = torch.load(self.pretrain)
                logger.info('Load pretrain model: %s', self.pretrain)
                if 'model' in model_data.keys():
                    model_data = model_data['model']
                if 'state_dict' in model_data.keys():
                    model_data = model_data['state_dict']
                model_data = {key.replace("module.", ""): value for key, value in model_data.items()}
                self.net.load_state_dict(model_data)
                self.is_pretrain = True
            else:
                logger.warning('Can not find the specific model %s, initial a new model...', self.pretrain)

        if self.ngpu > 1 and not self.hvd:
            self.net = torch.nn.DataParallel(self.net.to('cuda'), device_ids=self.devices)
        else:
            self.net.to('cuda')

        if self.rank == 0:
            logger.info('# of parameters: %d in model %s', count_parameters(self.net), self.net_name)

    def _build_optimizer(self):
        beta = 0.999
        momentum = 0.9
        if self.hvd:
            self.optimizer = Lamb(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr,
                                  betas=(momentum, beta))
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), self.lr,
                                              betas=(momentum, beta), amsgrad=True)

    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self._build_optimizer()

    def adjust_learning_rate(self, epoch):
        warmup = 0
        if epoch < warmup:
            warmup_total_iters = self.num_batches_per_epoch * warmup
            min_lr = self.lr / warmup_total_iters
            lr_interval = (self.lr - min_lr) / warmup_total_iters
            cur_lr = min_lr + lr_interval * self.train_iter
        else:
            if self.dataset.find('kitti') >= 0:
                # cur_lr = self.lr / (2**(epoch// 200))
                if epoch <= 300:
                    cur_lr = self.lr
                elif epoch <= 500:
                    cur_lr = self.lr * 0.1
                elif epoch <= 600:
                    cur_lr = self.lr * 0.01
                else:
                    cur_lr = self.lr * 0.001
                # cur_lr = self.lr
            elif self.dataset.find('sceneflow') >= 0:
                cur_lr = self.lr / (2 ** (epoch // 10))
            elif self.dataset.find('middlebury') >= 0 or self.dataset.find('rvc') >= 0:
                if epoch <= 1400:
                    cur_lr = self.lr
                elif epoch <= 1800:
                    cur_lr = self.lr * 0.1
                elif epoch <= 2200:
                    cur_lr = self.lr * 0.01
                else:
                    cur_lr = self.lr * 0.001
            else:
                cur_lr = self.lr / (2 ** (epoch // 150))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.current_lr = cur_lr
        return cur_lr

    def set_criterion(self, criterion):
        self.criterion = criterion

    def train_one_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        flow2_EPEs = AverageMeter()
        d1_metrics = AverageMeter()
        norm_EPEs = AverageMeter()
        angle_EPEs = AverageMeter()

        # switch to train mode
        self.net.train()
        end = time.time()
        cur_lr = self.adjust_learning_rate(epoch)
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))
        netconfig_setting = {}
        for i_batch, sample_batched in enumerate(self.train_loader):

            left_input = sample_batched['img_left'].to('cuda')
            right_input = sample_batched['img_right'].to('cuda')
            target_disp = sample_batched['gt_disp'].to('cuda')

            input_var = torch.cat((left_input, right_input), 1)

            data_time.update(time.time() - end)
            self.optimizer.zero_grad()
            netconf = '(netconf)'

            if self.net_name in ["fadnet", "ofa_fadnet", "mobilefadnet", 'slightfadnet', 'tinyfadnet', 'microfadnet',
                                 'xfadnet',
                                 'ucresnet']:

                subnet_seed = int('%d%.3d%.3d' % (epoch * self.num_batches_per_epoch + i_batch, i_batch, 0))
                random.seed(subnet_seed)
                ofa_settings = self.net.sample_active_subnet()
                # get sample net config
                for key, val in ofa_settings.items():
                    netconf = netconf + ', ' + key + ':{'
                    for ksval in val:
                        netconf += '{}'.format(ksval)
                    netconf += "}"

                output_net1, output_net2 = self.net(input_var)
                if self.net_name == 'ucresnet':
                    output_net1.reverse()
                    output_net2.reverse()
                loss_net1 = self.criterion(output_net1, target_disp)
                loss_net2 = self.criterion(output_net2, target_disp)
                loss = loss_net1 + loss_net2
                output_net2_final = output_net2[0]
                flow2_EPE = self.epe(output_net2_final.detach(), target_disp, maxdisp=self.maxdisp)
                d1m = d1_metric(output_net2_final, target_disp, maxdisp=self.maxdisp)
            elif self.net_name == "dispnetcs":
                output_net1, output_net2 = self.net(input_var)
                loss_net1 = self.criterion(output_net1, target_disp)
                loss_net2 = self.criterion(output_net2, target_disp)
                loss = loss_net1 + loss_net2
                output_net2_final = output_net2[0]
                flow2_EPE = self.epe(output_net2_final, target_disp)
            elif self.net_name == "dispnetcss":
                output_net1, output_net2, output_net3 = self.net(input_var)
                loss_net1 = self.criterion(output_net1, target_disp)
                loss_net2 = self.criterion(output_net2, target_disp)
                loss_net3 = self.criterion(output_net3, target_disp)
                loss = loss_net1 + loss_net2 + loss_net3
                output_net3_final = output_net3[0]
                flow2_EPE = self.epe(output_net3_final, target_disp, maxdisp=self.maxdisp)
                d1m = d1_metric(output_net3_final, target_disp, maxdisp=self.maxdisp)
            elif self.net_name == "psmnet" or self.net_name == "ganet":
                mask = target_disp < self.maxdisp
                mask.detach_()

                output1, output2, output3 = self.net(input_var)
                output1 = torch.unsqueeze(output1, 1)
                output2 = torch.unsqueeze(output2, 1)
                output3 = torch.unsqueeze(output3, 1)

                loss = 0.5 * F.smooth_l1_loss(output1[mask], target_disp[mask],
                                              size_average=True) + 0.7 * F.smooth_l1_loss(output2[mask],
                                                                                          target_disp[mask],
                                                                                          size_average=True) + F.smooth_l1_loss(
                    output3[mask], target_disp[mask], size_average=True)
                pred_disp = output3.detach()
                flow2_EPE = self.epe(pred_disp, target_disp, maxdisp=self.maxdisp)
                d1m = d1_metric(pred_disp, target_disp, maxdisp=self.maxdisp)
            elif self.net_name == "aanet":
                loss_weights = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]
                mask = target_disp < self.maxdisp
                mask.detach_()

                outputs = self.net(input_var)
                loss = 0.0
                for oid in range(len(outputs)):
                    output = torch.unsqueeze(outputs[oid], 1)
                    output = F.interpolate(output, size=(target_disp.size(-2), target_disp.size(-1)), mode='bilinear',
                                           align_corners=False) * (target_disp.size(-1) / output.size(-1))
                    loss += loss_weights[oid] * F.smooth_l1_loss(output[mask], target_disp[mask], size_average=True)
                pred_disp = torch.unsqueeze(outputs[-1].detach(), 1)
                flow2_EPE = self.epe(pred_disp, target_disp, maxdisp=self.maxdisp)
                d1m = d1_metric(pred_disp, target_disp, maxdisp=self.maxdisp)
            elif self.net_name == "gwcnet":
                mask = target_disp < self.maxdisp
                mask.detach_()

                output1, output2, output3, output4 = self.net(input_var)

                loss = 0.5 * F.smooth_l1_loss(output1[mask], target_disp[mask],
                                              size_average=True) + 0.5 * F.smooth_l1_loss(output2[mask],
                                                                                          target_disp[mask],
                                                                                          size_average=True) + 0.7 * F.smooth_l1_loss(
                    output3[mask], target_disp[mask], size_average=True) + F.smooth_l1_loss(output4[mask],
                                                                                            target_disp[mask],
                                                                                            size_average=True)
                flow2_EPE = self.epe(output3, target_disp)
                d1m = d1_metric(output3, target_disp, maxdisp=self.maxdisp)
            elif self.net_name == "ucnet":
                output = self.net(input_var)
                output.reverse()
                loss = self.criterion(output, target_disp)
                pred_disp = output[0]
                flow2_EPE = self.epe(pred_disp.detach(), target_disp, maxdisp=self.maxdisp)
                d1m = d1_metric(pred_disp, target_disp, maxdisp=self.maxdisp)
            else:
                output = self.net(input_var)
                loss = self.criterion(output, target_disp)
                if type(loss) is list or type(loss) is tuple:
                    loss = np.sum(loss)
                if type(output) is list or type(output) is tuple:
                    flow2_EPE = self.epe(output[0], target_disp)
                else:
                    flow2_EPE = self.epe(output, target_disp)

            # compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()

            # record loss and EPE
            losses.update(loss.detach(), input_var.size(0))
            flow2_EPEs.update(flow2_EPE, input_var.size(0))
            d1_metrics.update(d1m, input_var.size(0))
            # losses.update(4.0, input_var.size(0))
            # flow2_EPEs.update(4, input_var.size(0))
            # d1_metrics.update(4, input_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.train_iter += 1

            if self.rank == 0 and i_batch % 10 == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                            'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'
                            'D1-All {d1_metrics.val:.3f} ({d1_metrics.avg:.3f})\t'
                            '{netconf}\t'
                .format(
                    epoch, i_batch, self.num_batches_per_epoch, batch_time=batch_time,
                    data_time=data_time, loss=losses, flow2_EPE=flow2_EPEs, d1_metrics=d1_metrics, netconf=netconf))

            if flow2_EPEs.val < 10 and (i_batch + 1) >= self.num_batches_per_epoch * 0.99 - 1:
                netconfig_setting[flow2_EPEs.val] = ofa_settings
            # netconf = ''
        return losses.avg, flow2_EPEs.avg, netconfig_setting
        # return losses.avg, flow2_EPEs.avg

    def validate(self, config):
    # def validate(self):
        # end=0.

        print("subnet nums: {}".format(len(config)))
        nas = True
        # ks_lsit = [3, 5, 7]
        # depth_list = [2, 3, 4]
        # expand_list = [2, 4, 6, 8]
        # subnet_settings = []
        # ds = [4, 4, 4, 3, 3, 3, 2, 2, 2]
        # es = [8, 8, 8, 8, 8, 8, 8, 8, 8]
        # ks = [7, 5, 3, 7, 5, 3, 7, 5, 3]
        # for k in ks_lsit:
        #     for d in depth_list:
        #         for e in expand_list:
        #             subnet_settings.append([{
        #                 'ds': d,
        #                 'es': e,
        #                 'ks': k,
        #             }, 'K%s-D%s-E%s' % (k, d, e)])
        # for d, e, k in zip(ds, es, ks):
        #     subnet_settings.append([{
        #         'ds': d,
        #         'es': e,
        #         'ks': k,
        #     }, 'D%s-E%s-K%s' % (d, e, k)])
        batch_time = AverageMeter()
        if self.hvd:
            flow2_EPEs = HVDMetric('val_epe')
            d1_metrics = HVDMetric('val_d1')
        else:
            flow2_EPEs = AverageMeter()
            d1_metrics = AverageMeter()
        losses = AverageMeter()

        if nas:
            dict = config
            # for setting, name in subnet_settings:
            for keys, values in zip(dict.keys(), dict.values()):
                fad_settings = {}
                cunet_settings = {}
                count = 0
                netconf_str = ''
                for name, config in values.items():
                    netconf_str = netconf_str + ', ' + name + ':{'
                    for ksval in config:
                        netconf_str += '{}'.format(ksval)
                    netconf_str += "}"
                for key, value in zip(values.keys(), values.values()):
                    if count < 3:
                        fad_settings[key] = value
                    else:
                        cunet_settings[key] = value
                    count += 1

                self.net.set_active_subnet(**fad_settings)
                self.net.cunet.set_active_subnet(**cunet_settings)
                self.reset_running_statistics(self.net)
                # self.net.set_active_subnet(ks=[3,7,7,3,3,7,3,5,3,5,3,7], d=[3, 4, 3], e=[8,6,4,8,6,8,8,6,8,8,8,6])
                # self.net.cunet.set_active_subnet(ks=[5,5,5,5,7,3,5,3], d=[3, 4], e=[2,6,6,8,6,6,4,4])
                # ks:{377337353537}, es:{864868868886}, ds:{343}, cunet_ks:{55557353}, cunet_es:{26686644}, cunet_ds:{34}
                # data=self.data_provider.build_sub_train_loader(n_images, batch_size, num_worker, num_replicas, rank)
                # set_running_statistics(self.net,data)
                self.net.eval()
                end = time.time()
                for i, sample_batched in enumerate(self.test_loader):
                    left_input = sample_batched['img_left'].to('cuda')
                    right_input = sample_batched['img_right'].to('cuda')
                    left_input = F.interpolate(left_input, self.scale_size, mode='bilinear')
                    right_input = F.interpolate(right_input, self.scale_size, mode='bilinear')
                    input_var = torch.cat((left_input, right_input), 1)

                    target_disp = sample_batched['gt_disp']
                    target_disp = target_disp.to('cuda')
                    target_disp = torch.autograd.Variable(target_disp, requires_grad=False)

                    with torch.no_grad():
                        output_net1, output_net2 = self.net(input_var)
                    output_net1 = scale_disp(output_net1, (output_net1.size()[0], self.img_height, self.img_width))
                    output_net2 = scale_disp(output_net2, (output_net2.size()[0], self.img_height, self.img_width))

                    loss_net1 = self.epe(output_net1, target_disp, maxdisp=self.maxdisp)
                    loss_net2 = self.epe(output_net2, target_disp, maxdisp=self.maxdisp)
                    loss = loss_net1 + loss_net2
                    flow2_EPE = self.epe(output_net2, target_disp, maxdisp=self.maxdisp)
                    d1m = d1_metric(output_net2, target_disp, maxdisp=self.maxdisp)

                    losses.update(loss, input_var.size(0))
                    flow2_EPEs.update(flow2_EPE, input_var.size(0))
                    d1_metrics.update(d1m, input_var.size(0))

                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % 10 == 0:
                        # logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}\t D1-All {4}\t'
                        logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}\t D1-All {4}\t Config {5}\t Train EPE: {6}'
                                    .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val, d1_metrics.val,
                                            netconf_str, keys))
                    # ))
        # switch to evaluate mode
        else:
            self.net.eval()
            end = time.time()
            for i, sample_batched in enumerate(self.test_loader):

                left_input = sample_batched['img_left'].to('cuda')
                right_input = sample_batched['img_right'].to('cuda')
                left_input = F.interpolate(left_input, self.scale_size, mode='bilinear')
                right_input = F.interpolate(right_input, self.scale_size, mode='bilinear')
                input_var = torch.cat((left_input, right_input), 1)

                target_disp = sample_batched['gt_disp']
                target_disp = target_disp.to('cuda')
                target_disp = torch.autograd.Variable(target_disp, requires_grad=False)

                if self.net_name in ['fadnet', 'ofa_fadnet', 'mobilefadnet', 'slightfadnet', 'tinyfadnet',
                                     'microfadnet', 'xfadnet']:
                    with torch.no_grad():
                        output_net1, output_net2 = self.net(input_var)
                    output_net1 = scale_disp(output_net1, (output_net1.size()[0], self.img_height, self.img_width))
                    output_net2 = scale_disp(output_net2, (output_net2.size()[0], self.img_height, self.img_width))

                    loss_net1 = self.epe(output_net1, target_disp, maxdisp=self.maxdisp)
                    loss_net2 = self.epe(output_net2, target_disp, maxdisp=self.maxdisp)
                    loss = loss_net1 + loss_net2
                    flow2_EPE = self.epe(output_net2, target_disp, maxdisp=self.maxdisp)
                    d1m = d1_metric(output_net2, target_disp, maxdisp=self.maxdisp)
                elif self.net_name in ['psmnet', 'ganet', 'gwcnet', 'aanet']:
                    with torch.no_grad():
                        output_net3 = self.net(input_var)
                    if self.net_name == 'aanet':
                        output_net3 = output_net3[-1]
                    if output_net3.dim() == 3:
                        output_net3 = output_net3.unsqueeze(1)
                    output_net3 = scale_disp(output_net3, (output_net3.size()[0], 540, 960))
                    # output_net3 = scale_disp(output_net3, (output_net3.size()[0], 436, 1024))
                    loss = self.epe(output_net3, target_disp, maxdisp=self.maxdisp)
                    flow2_EPE = loss
                    d1m = d1_metric(output_net3, target_disp, maxdisp=self.maxdisp)
                elif self.net_name == 'dispnetcss':
                    output_net1, output_net2, output_net3 = self.net(input_var)
                    output_net1 = scale_disp(output_net1, (output_net1.size()[0], self.img_height, self.img_width))
                    output_net2 = scale_disp(output_net2, (output_net2.size()[0], self.img_height, self.img_width))
                    output_net3 = scale_disp(output_net3, (output_net3.size()[0], self.img_height, self.img_width))

                    loss_net1 = self.epe(output_net1, target_disp, maxdisp=self.maxdisp)
                    loss_net2 = self.epe(output_net2, target_disp, maxdisp=self.maxdisp)
                    loss_net3 = self.epe(output_net3, target_disp, maxdisp=self.maxdisp)
                    loss = loss_net1 + loss_net2 + loss_net3
                    flow2_EPE = self.epe(output_net3, target_disp, maxdisp=self.maxdisp)
                    d1m = d1_metric(output_net3, target_disp, maxdisp=self.maxdisp)
                elif self.net_name == 'ucnet':
                    with torch.no_grad():
                        output = self.net(input_var)[-1]
                    output = scale_disp(output, (output.size()[0], self.img_height, self.img_width))

                    loss = self.epe(output, target_disp, maxdisp=self.maxdisp)
                    flow2_EPE = self.epe(output, target_disp, maxdisp=self.maxdisp)
                    d1m = d1_metric(output, target_disp, maxdisp=self.maxdisp)
                elif self.net_name == 'ucresnet':
                    with torch.no_grad():
                        outputs_net1, outputs_net2 = self.net(input_var)
                    output_net1 = outputs_net1[-1]
                    output_net2 = outputs_net2[-1]
                    output_net1 = scale_disp(output_net1, (output_net1.size()[0], self.img_height, self.img_width))
                    output_net2 = scale_disp(output_net2, (output_net2.size()[0], self.img_height, self.img_width))

                    loss_net1 = self.epe(output_net1, target_disp, maxdisp=self.maxdisp)
                    loss_net2 = self.epe(output_net2, target_disp, maxdisp=self.maxdisp)
                    loss = loss_net1 + loss_net2
                    flow2_EPE = self.epe(output_net2, target_disp, maxdisp=self.maxdisp)
                    d1m = d1_metric(output_net2, target_disp, maxdisp=self.maxdisp)
                else:
                    output = self.net(input_var)
                    output_net1 = output[0]
                    # output_net1 = output_net1.squeeze(1)
                    # print(output_net1.size())
                    output_net1 = scale_disp(output_net1, (output_net1.size()[0], self.img_height, self.img_width))
                    loss = self.epe(output_net1, target_disp)
                    flow2_EPE = self.epe(output_net1, target_disp)

                # if type(loss) is list or type(loss) is tuple:
                #    loss = loss[0]
                # if type(output) is list or type(output_net1) :
                #    flow2_EPE = self.epe(output[0], target_disp)
                # else:
                #    flow2_EPE = self.epe(output, target_disp)

                # record loss and EPE
                losses.update(loss, input_var.size(0))
                flow2_EPEs.update(flow2_EPE, input_var.size(0))
                d1_metrics.update(d1m, input_var.size(0))
                # if loss.data.item() == loss.data.item():
                #    losses.update(loss.data.item(), input_var.size(0))
                # if flow2_EPE.data.item() == flow2_EPE.data.item():
                #    if self.hvd:
                #        flow2_EPEs.update(flow2_EPE, input_var.size(0))
                #    else:
                #        flow2_EPEs.update(flow2_EPE.data.item(), input_var.size(0))
                # if d1m.data.item() == d1m.data.item():
                #    if self.hvd:
                #        d1_metrics.update(d1m.data, input_var.size(0))
                #    else:
                #        d1_metrics.update(d1m.data.item(), input_var.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}\t D1-All {4}'
                                .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val, d1_metrics.val))

        logger.info(' * EPE {:.3f}, D1-All {:.3f}'.format(flow2_EPEs.avg, d1_metrics.avg))
        return flow2_EPEs.avg

    def get_model(self):
        return self.net.state_dict()

    def reset_running_statistics(self, net=None, subset_size=200, subset_batch_size=20, data_loader=None):
        if net is None:
            net = self.net
        if data_loader is None:
            data_loader = self.build_sub_train_loader(subset_size, subset_batch_size)
        set_running_statistics(net, data_loader)

    def build_sub_train_loader(self, n_images, batch_size):
        train_dataset = SceneFlowDataset(txt_file=self.trainlist, root_dir=self.datapath, phase='train')
        n_samples = len(train_dataset)
        g = torch.Generator()
        SUB_SEED = 937162211
        g.manual_seed(SUB_SEED)
        rand_indexes = torch.randperm(n_samples, generator=g).tolist()
        chosen_indexes = rand_indexes[:n_images]
        sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
        new_train_dataset = train_dataset
        sub_data_loader = torch.utils.data.DataLoader(new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                                                      num_workers=4, pin_memory=True)
        return sub_data_loader
