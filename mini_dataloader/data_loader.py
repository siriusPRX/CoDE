import os
import numpy as np
import random
from .func import read, train_process, test_process
from .get_data import get_data

def np_normalize(inputs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x = inputs.copy()
    x[0] = (x[0] - mean[0]) / std[0]
    x[1] = (x[1] - mean[1]) / std[1]
    x[2] = (x[2] - mean[2]) / std[2]
    return x

class DataLoader():
    def __init__(self, name, mode='train', ratio=0.7, batch_size=32, input_size=512, in_channels=3,
                 divide_shuffle=False, augment_prob=0.5, item_shuffle=True, normalize=False, enable_aug_types=[0, 1, 2],
                 intensity={'rates': None, 'qfs': None, 'sds': None, 'ksizes': None, 'ratios': None}):
        root = '/dataset'
        self.batch_size = batch_size
        self.input_size = input_size
        self.enable_aug_types = enable_aug_types
        self.intensity = intensity
        self.in_channels = in_channels
        self.augment_prob = augment_prob
        self.item_shuffle = item_shuffle
        self.normalize = normalize
        self.indices = 0    # The current read position of the dataset
        self.prefix, self.flist = get_data(root=root, name=name, ratio=ratio, divide_shuffle=divide_shuffle, mode=mode)
        print('Mode: {} , Dataset: name:{} | nums: {}'.format(mode, self.prefix, len(self.flist)))

    def reset(self):
        self.indices = 0

    def get_item(self):
        # _/_/_/ get a batch of data _/_/_/ #
        selected_flist = self.flist[self.indices:self.indices + self.batch_size]
        if (self.indices + self.batch_size) > len(self.flist):
            supplement = (self.indices + self.batch_size) % len(self.flist)
            self.indices = 0
            if supplement > 0:
                selected_flist += self.flist[:supplement]
                if self.item_shuffle:
                    random.shuffle(self.flist)
        else:
            self.indices += self.batch_size

        # _/_/_/ processing a batch of data _/_/_/ #
        forgeris = np.zeros((self.batch_size, self.in_channels, self.input_size, self.input_size),
                            dtype=np.float32)
        masks = np.zeros((self.batch_size, 1, self.input_size, self.input_size), dtype=np.float32)
        for idx, path_list in enumerate(selected_flist):
            if ',' in path_list:
                forgery_path = os.path.join(self.prefix, path_list.split(',')[-2])
                mask_path = os.path.join(self.prefix, path_list.split(',')[-1])
            else:
                forgery_path = os.path.join(self.prefix, path_list.split(' ')[-2])
                mask_path = os.path.join(self.prefix, path_list.split(' ')[-1])
            forgery, mask = read(forgery_path, mask_path)
            forgery, mask = train_process(forgery, mask, augment_prob=self.augment_prob, input_size=self.input_size,
                                          enable_aug_types=self.enable_aug_types, intensity=self.intensity)
            if self.normalize:
                forgery = np_normalize(forgery, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            forgeris[idx], masks[idx] = forgery, mask

        return forgeris, masks

    def test_get_item(self, idx):
        # _/_/_/ processing single data according to index _/_/_/ #
        forgeris, masks = list(), list()
        if ',' in self.flist[idx]:
            forgery_path = os.path.join(self.prefix, self.flist[idx].split(',')[-2])
            mask_path = os.path.join(self.prefix, self.flist[idx].split(',')[-1])
        else:
            forgery_path = os.path.join(self.prefix, self.flist[idx].split(' ')[-2])
            mask_path = os.path.join(self.prefix, self.flist[idx].split(' ')[-1])
        forgery, gt_mask = read(forgery_path, mask_path)
        forgery, mask = test_process(forgery, gt_mask, input_size=self.input_size, augment_prob=self.augment_prob,
                                     enable_aug_types=self.enable_aug_types, intensity=self.intensity)
        forgeris.append(forgery)
        masks.append(mask)
        return (np.stack(forgeris), np.stack(masks),
                os.path.splitext(os.path.basename(forgery_path))[0],
                os.path.splitext(os.path.basename(mask_path))[0])


