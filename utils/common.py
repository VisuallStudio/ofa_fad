import json, yaml
import logging
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


# Add by jingbo
def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def min_divisible_value(n1, v1):
    """ make sure v1 is divisible by n1, otherwise decrease v1 """
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def get_net_device(net):
    return net.parameters().__next__().device


def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func == 'LeakyRelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_func is None or act_func == 'none':
        return None
    else:
        raise ValueError('do not support: %s' % act_func)


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hsigmoid()'


def load_loss_scheme(loss_config):
    with open(loss_config, 'r') as f:
        loss_json = yaml.safe_load(f)

    return loss_json


DEBUG = 0
logger = logging.getLogger()

if DEBUG:
    # coloredlogs.install(level='DEBUG')
    logger.setLevel(logging.DEBUG)
else:
    # coloredlogs.install(level='INFO')
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)


# from netdef_slim.utils.io import read
# left_img = sys.argv[1]
# subfolder = sys.argv[2]
#
# occ_file = 'tmp/disp.L.float3'
# occ_data = read(occ_file) # returns a numpy array
#
# import matplotlib.pyplot as plt
# occ_data = occ_data[::-1, :, :] * -1.0
# print(np.mean(occ_data))
##plt.imshow(occ_data[:,:,0], cmap='gray')
## plt.show()
#
# subfolder = "detect_results/%s" % subfolder
# if not os.path.exists(subfolder):
#    os.makedirs(subfolder)
#
##name_items = left_img.split('.')[0].split('/')
##save_name = '_'.join(name_items) + '.pfm'
# name_items = left_img.split('/')
# filename = name_items[-1]
# topfolder = name_items[-2]
# save_name = filename + '.pfm'
# target_folder = '%s/%s' % (subfolder, topfolder)
# print('target_folder: ', target_folder)
# if not os.path.exists(target_folder):
#    os.makedirs(target_folder)
# save_pfm('%s/%s' % (target_folder, save_name), occ_data[:,:,0])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class DistributedTensor(object):

    def __init__(self, name):
        self.name = name
        self.sum = None
        self.count = torch.zeros(1)[0]
        self.synced = False

    def update(self, val, delta_n=1):
        val *= delta_n
        if self.sum is None:
            self.sum = val.detach()
        else:
            self.sum += val.detach()
        self.count += delta_n

    @property
    def avg(self):
        import horovod.torch as hvd
        if not self.synced:
            self.sum = hvd.allreduce(self.sum, name=self.name)
            self.synced = True
        return self.sum / self.count