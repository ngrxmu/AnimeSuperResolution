'''
preprocess the data and create the dataset
'''

from torch.utils import data
import random
import torch
import h5py


class MyDataSet(data.Dataset):
    def __init__(self, h5root, lr_size=64, scale=4):
        super(MyDataSet, self).__init__()
        self.h5root = h5root
        self.lr_size = lr_size
        self.scale = scale

    def random_crop(self, lr, hr):
        lr_x1 = random.randint(0, lr.shape[2]-self.lr_size)
        lr_x2 = lr_x1+self.lr_size
        lr_y1 = random.randint(0, lr.shape[1]-self.lr_size)
        lr_y2 = lr_y1+self.lr_size
        hr_x1 = lr_x1 * self.scale
        hr_x2 = lr_x2 * self.scale
        hr_y1 = lr_y1 * self.scale
        hr_y2 = lr_y2 * self.scale
        lr = lr[:, lr_y1:lr_y2, lr_x1:lr_x2]
        hr = hr[:, hr_y1:hr_y2, hr_x1:hr_x2]
        return lr, hr

    def random_horizontal_flip(self, lr, hr):
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        return lr, hr

    def random_vertical_flip(self, lr, hr):
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        return lr, hr

    def random_rotation(self, lr, hr):
        if random.random() < 0.5:
            lr = torch.rot90(lr, dims=(2, 1))
            hr = torch.rot90(hr, dims=(2, 1))
        return lr, hr

    def __getitem__(self, index):
        with h5py.File(self.h5root, 'r') as f:
            hr = torch.from_numpy(f['hr'][str(index)][::])
            lr = torch.from_numpy(f['lr'][str(index)][::])
            lr, hr = self.random_crop(lr, hr)
            lr, hr = self.random_vertical_flip(lr, hr)
            lr, hr = self.random_horizontal_flip(lr, hr)
            lr, hr = self.random_rotation(lr, hr)
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5root, 'r') as f:
            return len(f['hr'])
