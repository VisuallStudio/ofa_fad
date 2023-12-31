from __future__ import print_function
import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import skimage
import torch.cuda as ct
from net_builder import SUPPORT_NETS, build_net
from losses.multiscaleloss import multiscaleloss
import torch.nn.functional as F
import torch.nn as nn
from dataloader.StereoLoader import StereoDataset
from utils.preprocess import scale_disp, save_pfm
from utils.common import count_parameters
from torch.utils.data import DataLoader
from torchvision import transforms
import psutil
# from torch2trt import torch2trt
from networks.submodules import build_corr, channel_length

process = psutil.Process(os.getpid())
cudnn.benchmark = True


def detect(opt):
    net_name = opt.net
    model = opt.model
    result_path = opt.rp
    file_list = opt.filelist
    filepath = opt.filepath

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    devices = [int(item) for item in opt.devices.split(',')]
    ngpu = len(devices)

    # build net according to the net name
    if net_name == "psmnet" or net_name == "ganet":
        net = build_net(net_name)(192)
    elif net_name in ["fadnet", "ofa_fadnet", "dispnetc"]:
        net = build_net(net_name)(ks_list=[3, 5, 7], expand_ratio_list=8, depth_list=4)
    elif net_name == "mobilefadnet":
        # B, max_disp, H, W = (wopt.batchSize, 40, 72, 120)
        shape = (opt.batchSize, 40, 72, 120)  # TODO: Should consider how to dynamically use
        warp_size = (opt.batchSize, 3, 576, 960)
        net = build_net(net_name)(batchNorm=False, lastRelu=True, input_img_shape=shape, warp_size=warp_size)

    model_data = torch.load(model)
    print(model_data.keys())
    if 'state_dict' in model_data.keys():
        state_dict = {key.replace("module.", ""): value for key, value in model_data['state_dict'].items()}
        net.load_state_dict(state_dict)
    else:
        state_dict = {key.replace("module.", ""): value for key, value in model_data.items()}
        net.load_state_dict(state_dict)

    # net.re_organize_middle_weights(expand_ratio_stage=2)

    net.set_active_subnet(ks=[5,3,5,3,7,5,3,3,5,7,3,7], d=4, e=8)
    net.cunet.set_active_subnet(ks=[5,5,5,5,7,5,3,5], d=4, e=8)
    # ks:{535375335737}, es:{888888888888}, ds:{444}, cunet_ks:{55557535}, cunet_es:{88888888}, cunet_ds:{44}
    # net.get_active_subnet(preserve_weight=True)
    if ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=devices)

    num_of_parameters = count_parameters(net)
    print('Model: %s, # of parameters: %d' % (net_name, num_of_parameters))

    batch_size = int(opt.batchSize)
    test_dataset = StereoDataset(txt_file=file_list, root_dir=filepath, phase='detect')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, \
                             shuffle=False, num_workers=1, \
                             pin_memory=False)

    net.eval()
    net = net.cuda()
    # print(next(net.parameters()).is_cuda)  true

    # for i, sample_batched in enumerate(test_loader):
    #    input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)
    #    num_of_samples = input.size(0)
    #    input = input.cuda()
    #    x = input
    #    break

    s = time.time()

    avg_time = []
    display = 50
    warmup = 12
    for i, sample_batched in enumerate(test_loader):
        # if i > 215:
        #    break
        stime = time.time()

        input = torch.cat((sample_batched['img_left'], sample_batched['img_right']), 1)

        print('input Shape: {}'.format(input.size()))
        num_of_samples = input.size(0)

        # output, input_var = detect_batch(net, sample_batched, opt.net, (540, 960))

        input = input.cuda()
        input_var = input  # torch.autograd.Variable(input, volatile=True)
        input_var = F.interpolate(input_var, (576, 960), mode='bilinear')
        iotime = time.time()
        print('[{}] IO time:{}'.format(i, iotime - stime))

        if i > warmup:
            ss = time.time()

        with torch.no_grad():
            if opt.net == "psmnet" or opt.net == "ganet":
                output = net(input_var)
                output = output.unsqueeze(1)
            elif opt.net == "dispnetc":
                output = net(input_var)[0]
            else:
                output = net(input_var)[-1]
        itime = time.time()
        print('[{}] Inference time:{}'.format(i, itime - iotime))

        if i > warmup:
            avg_time.append((time.time() - ss))
            if (i - warmup) % display == 0:
                print('Average inference time: %f' % np.mean(avg_time))
                mbytes = 1024. * 1024
                print(
                    'GPU memory usage memory_allocated: %d MBytes, max_memory_allocated: %d MBytes, memory_cached: %d MBytes, max_memory_cached: %d MBytes, CPU memory usage: %d MBytes' % \
                    (ct.memory_allocated() / mbytes, ct.max_memory_allocated() / mbytes, ct.memory_cached() / mbytes,
                     ct.max_memory_cached() / mbytes, process.memory_info().rss / mbytes))
                avg_time = []

        print('[%d] output shape:' % i, output.size())
        output = scale_disp(output, (output.size()[0], 540, 960))
        disp = output[:, 0, :, :]
        ptime = time.time()
        print('[{}] Post-processing time:{}'.format(i, ptime - itime))

        for j in range(num_of_samples):

            name_items = sample_batched['img_names'][0][j].split('/')
            # write disparity to file
            output_disp = disp[j]
            np_disp = disp[j].float().cpu().numpy()

            print('Batch[{}]: {}, average disp: {}({}-{}).'.format(i, j, np.mean(np_disp), np.min(np_disp),
                                                                   np.max(np_disp)))

            if opt.format == 'png':
                save_name = '_'.join(name_items).replace(".png", "_d.png")  # for girl02 dataset
                print('Name: {}'.format(save_name))
                skimage.io.imsave(os.path.join(result_path, save_name), (np_disp * 256).astype('uint16'))
            elif opt.format == 'pfm':
                save_name = '_'.join(name_items).replace("png", "pfm")  # for girl02 dataset
                print('Name: {}'.format(save_name))
                np_disp = np.flip(np_disp, axis=0)
                save_pfm('{}/{}'.format(result_path, save_name), np_disp)

        print('Current batch time used:: {}'.format(time.time() - stime))

    print('Evaluation time used: {}'.format(time.time() - s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='fadnet', choices=SUPPORT_NETS)
    parser.add_argument('--model', type=str, help='model to load', default='best.pth')
    parser.add_argument('--filelist', type=str, help='file list', default='FlyingThings3D_release_TEST.list')
    parser.add_argument('--format', type=str, help='output disparityformat', default='pfm', choices=['pfm', 'png'])
    parser.add_argument('--filepath', type=str, help='file path', default='./data')
    parser.add_argument('--devices', type=str, help='devices', default='0')
    parser.add_argument('--display', type=int, help='Num of samples to print', default=10)
    parser.add_argument('--rp', type=str, help='result path', default='./result')
    parser.add_argument('--flowDiv', type=float, help='flow division', default='1.0')
    parser.add_argument('--batchSize', type=int, help='mini batch size', default=1)

    opt = parser.parse_args()
    detect(opt)
