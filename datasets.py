#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8

import os
import torch
from PIL import Image
from torch.utils import data
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
import cv2
import csv
import random
from red0orange.file import get_image_files


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    # (left,right,up,down)分别对应四个pad
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    # torch自带的pad函数
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


class ClsDataset(data.Dataset):
    def __init__(self, csv_path, size=224):
        self.data = []
        self.size = size
        if not (isinstance(csv_path, list) or isinstance(csv_path, tuple)):
            csv_path_list = [csv_path]
        else:
            csv_path_list = csv_path

        for per_csv_path in csv_path_list:
            with open(per_csv_path, 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    if len(line) == 0:
                        continue
                    self.data.append([line[1], int(line[0])])
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        # print(img_path)
        img = Image.open(img_path)
        img_tensor = T.ToTensor()(img)
        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3, :, :]
        pad_img_tensor, _ = pad_to_square(img_tensor, 0)
        pad_img_tensor = T.Resize(self.size)(pad_img_tensor)
        return img_path, pad_img_tensor, label


class DirectClsDataset(data.Dataset):
    def __init__(self, data, size=224, shuffle=False):
        self.data = data
        self.size = size
        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        # print(img_path)
        img = Image.open(img_path)
        img_tensor = T.ToTensor()(img)
        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3, :, :]
        pad_img_tensor, _ = pad_to_square(img_tensor, 0)
        pad_img_tensor = T.Resize(self.size)(pad_img_tensor)
        return img_path, pad_img_tensor, label

