from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from utils.preprocess import *
from torchvision import transforms
import time
from dataloader.EXRloader import load_exr
import torch.nn.functional as F
from dataloader.commons import normalize_method

class SceneFlowDataset(Dataset):

    def __init__(self, txt_file, root_dir, phase='train', load_disp=True, load_norm=False, to_angle=False, scale_size=(576, 960), normalize=normalize_method):
        """
        Args:
            txt_file [string]: Path to the image list
            transform (callable, optional): Optional transform to be applied                on a sample
        """
        with open(txt_file, "r") as f:
            self.imgPairs = np.array(f.readlines())

        self.root_dir = root_dir
        self.phase = phase
        self.load_disp = load_disp
        self.load_norm = load_norm
        self.to_angle = to_angle
        self.scale_size = scale_size
        self.img_size = (540, 960)

        self.normalize = normalize

    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size

    def __len__(self):
        return len(self.imgPairs)

    def __getitem__(self, idx):

        img_names = self.imgPairs[idx].rstrip().split()

        img_left_name = os.path.join(self.root_dir, img_names[0])
        img_right_name = os.path.join(self.root_dir, img_names[1])
        if self.load_disp:
            gt_disp_name = os.path.join(self.root_dir, img_names[2])
        if self.load_norm:
            gt_norm_name = os.path.join(self.root_dir, img_names[3])

        def load_rgb(filename):

            img = None
            if filename.find('.npy') > 0:
                img = np.load(filename)
            else:
                img = Image.open(filename)
                img = np.array(img)
                #img = io.imread(filename)
                if len(img.shape) == 2:
                    img = img[:,:,np.newaxis]
                    img = np.pad(img, ((0, 0), (0, 0), (0, 2)), 'constant')
                    img[:,:,1] = img[:,:,0]
                    img[:,:,2] = img[:,:,0]
                h, w, c = img.shape
                if c == 4:
                    img = img[:,:,:3]
            return img
           
        def load_disp(filename):
            gt_disp = None
            if gt_disp_name.endswith('pfm'):
                gt_disp, scale = load_pfm(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            elif gt_disp_name.endswith('npy'):
                gt_disp = np.load(gt_disp_name)
                gt_disp = gt_disp[::-1, :]
            elif gt_disp_name.endswith('exr'):
                gt_disp = load_exr(filename)
            else:
                gt_disp = Image.open(gt_disp_name)
                gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256

            return gt_disp

        def load_norm(filename):
            gt_norm = None
            if filename.endswith('exr'):
                gt_norm = load_exr(filename)
                
                # transform visualization normal to its true value
                gt_norm = gt_norm * 2.0 - 1.0

                ## fix opposite normal
                #m = gt_norm >= 0
                #m[:,:,0] = False
                #m[:,:,1] = False
                #gt_norm[m] = - gt_norm[m]

            return gt_norm

        s = time.time()
        left = load_rgb(img_left_name)
        right = load_rgb(img_right_name)
        if self.load_disp:
            gt_disp = load_disp(gt_disp_name)
        if self.load_norm:
            gt_norm = load_norm(gt_norm_name)

        s = time.time()

        h, w, _ = left.shape
        # normalize with imagenet stats and tensor()
        if self.normalize == 'imagenet':
            if self.phase == 'detect' or self.phase == 'test':
                rgb_transform = default_transform()
            else:
                rgb_transform = inception_color_preproccess()
            img_left = rgb_transform(left)
            img_right = rgb_transform(right)
        # instance normalization
        else:
            img_left = np.zeros([3, h, w], 'float32')
            img_right = np.zeros([3, h, w], 'float32')
            for c in range(3):
                img_left[c, :, :] = (left[:, :, c] - np.mean(left[:, :, c])) / np.std(left[:, :, c])
                img_right[c, :, :] = (right[:, :, c] - np.mean(right[:, :, c])) / np.std(right[:, :, c])

        if self.load_disp:
            gt_disp = gt_disp[np.newaxis, :]
            gt_disp = torch.from_numpy(gt_disp.copy()).float()

        if self.load_norm:
            gt_norm = gt_norm.transpose([2, 0, 1])
            gt_norm = torch.from_numpy(gt_norm.copy()).float()

        if self.phase == 'train':

            h, w = img_left.shape[1:3]
            th, tw = 384, 768
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)

            img_left = img_left[:, top: top + th, left: left + tw]
            img_right = img_right[:, top: top + th, left: left + tw]
            if self.load_disp:
                gt_disp = gt_disp[:, top: top + th, left: left + tw]
            if self.load_norm:
                gt_norm = gt_norm[:, top: top + th, left: left + tw]
    
        if self.to_angle:
            norm_size = gt_norm.size()
            gt_angle = torch.empty(2, norm_size[1], norm_size[2], dtype=torch.float)
            gt_angle[0, :, :] = torch.atan(gt_norm[0, :, :] / gt_norm[2, :, :])
            gt_angle[1, :, :] = torch.atan(gt_norm[1, :, :] / gt_norm[2, :, :])
 

        sample = {  'img_left': img_left, 
                    'img_right': img_right, 
                    'img_names': img_names
                 }

        if self.load_disp:
            sample['gt_disp'] = gt_disp
        if self.load_norm:
            if self.to_angle:
                sample['gt_angle'] = gt_angle
            else:
                sample['gt_norm'] = gt_norm

        return sample

    # def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
    #     # used for resetting BN running statistics
    #     #if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
    #     if self.__dict__.get('sub_train_list', None) is None:
    #         if num_worker is None:
    #             num_worker = 4
    #
    #         n_samples = len(self.train.dataset)
    #         g = torch.Generator()
    #         g.manual_seed(DataProvider.SUB_SEED)
    #         rand_indexes = torch.randperm(n_samples, generator=g).tolist()
    #
    #         new_train_dataset = self.train_dataset(
    #             self.build_train_transform(print_log=False))
    #         chosen_indexes = rand_indexes[:n_images]
    #         if num_replicas is not None:
    #             sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, True, np.array(chosen_indexes))
    #         else:
    #             sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    #         sub_data_loader = torch.utils.data.DataLoader(
    #             new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
    #             num_workers=num_worker, pin_memory=True,
    #         )
    #         self.__dict__['sub_train_list'] = []
    #         for sample in sub_data_loader:
    #             self.__dict__['sub_train_list'].append(sample)
    #     #return self.__dict__['sub_train_%d' % self.active_img_size]
    #     return self.__dict__['sub_train_list']