#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
import os
import torch
import shutil
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import ClsDataset
from tqdm import tqdm
import numpy as np

from red0orange.utils import *
import pretrainedmodels


if __name__ == '__main__':
    device = "cuda"
    batch_size = 36
    valid_csv_path = "valid.csv"
    checkout_path = ""
    save_path = ""

    valid_data = ClsDataset(valid_csv_path)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    # model = LeNet5(option.num_classes)
    model_name = 'resnet18'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    if checkout_path:
        model.load_state_dict(torch.load(checkout_path))

    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter("loss")
    acc_meter = AccMeter("acc")
    for ii, (image_paths, images, labels) in tqdm(enumerate(valid_dataloader)):
        input = images.to(device)
        target = labels.to(device)
        output = model(input)
        loss = F.cross_entropy(output, target)
        target = target.cpu().numpy()
        predict = torch.argmax(output, dim=1).cpu().numpy()

        loss_meter.update(loss.detach().item())
        acc_meter.update(np.sum(target == predict), predict.shape[0])
    print('correct: {}'.format(acc_meter.acc))
    print('loss: {}'.format(loss_meter.avg))
