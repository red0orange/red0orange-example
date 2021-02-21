#!/home/hdh3/anaconda3/bin/python
# encoding: utf-8
import os
import torch
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import ClsDataset
from tqdm import tqdm
import numpy as np

from red0orange.args import option, BaseOption, BaseParam
from red0orange.recorder import recorder
from red0orange.database import exp_result
from red0orange.utils import *
import pretrainedmodels


class TrainOption(BaseOption):
    # base
    n_epoch = BaseParam(int, 50)
    lr = BaseParam(float, 0.01)
    batch_size = BaseParam(int, 36)
    opt = BaseParam(str, "adam")
    num_classes = BaseParam(int, 2)
    valid_every = BaseParam(int, 1)

    checkpoint_path = BaseParam(str, '')
    train_csv_path = BaseParam(str, "cls_train.csv")
    valid_csv_path = BaseParam(str, "cls_valid.csv")
    checkpoint_save_path = BaseParam(str, 'checkout_point')
    num_workers = BaseParam(int, 4)
    device = BaseParam(str, 'cuda')


if __name__ == '__main__':
    option.init(TrainOption)
    recorder.init(option.save_root)

    train_data = ClsDataset(option.train_csv_path)
    train_dataloader = DataLoader(train_data, batch_size=option.batch_size, shuffle=False)
    valid_data = ClsDataset(option.valid_csv_path)
    valid_dataloader = DataLoader(valid_data, batch_size=option.batch_size, shuffle=False)

    # model = LeNet5(option.num_classes)
    model_name = 'resnet18'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    if option.checkpoint_path:
        model.load_state_dict(torch.load(option.checkpoint_path))
    stats_names = ['err_rate', 'max_conf', 'loss', 'reg']

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=option.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=option.lr, steps_per_epoch=len(train_dataloader), epochs=option.n_epoch)

    model = model.to(option.device)
    for epoch in range(0, option.n_epoch):
        # train
        model.train()

        loss_meter = AverageMeter("loss")
        for ii, train_data in tqdm(enumerate(train_dataloader)):
            image_paths, images, labels = train_data

            input = images.to(option.device)
            target = labels.to(option.device)
            output = model(input)

            loss = criterion(output, target)
            loss = loss * ((target * 10) + 1)
            loss = torch.sum(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            loss_meter.update(loss.detach().item())
        recorder.msg('train_loss: {}'.format(loss_meter.avg))
        recorder.metric('train', {
            "loss": loss_meter.avg,
        })

        # valid
        if epoch % option.valid_every == 0:
            model.eval()
            loss_meter = AverageMeter("loss")
            acc_meter = AccMeter("acc")
            for ii, (_, images, labels) in tqdm(enumerate(valid_dataloader)):
                input = images.to(option.device)
                target = labels.to(option.device)
                output = model(input)
                loss = F.cross_entropy(output, target)
                target = target.cpu().numpy()
                predict = torch.argmax(output, dim=1).cpu().numpy()

                loss_meter.update(loss.detach().item())
                acc_meter.update(np.sum(target == predict), predict.shape[0])
            recorder.msg('correct: {}'.format(acc_meter.acc))
            recorder.msg('loss: {}'.format(loss_meter.avg))
            recorder.metric('val', {
                "loss": loss_meter.avg,
                "acc" : acc_meter.acc,
            })
            recorder.save_model(model.state_dict(), epoch)
