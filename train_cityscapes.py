import torch
import torch.nn as nn
import numpy as np
import pickle
import deeplab_resnet
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import matplotlib.pyplot as plt
from tqdm import *
import random
from docopt import docopt
from scipy.io import loadmat
from matplotlib import gridspec
from utils import *
from collections import OrderedDict
import dataset
from tensorboardX import SummaryWriter
from cityscapes import *
import vis
import time

docstr = """
Train Deeplabv2 on Cityscapes

Usage: 
    train_cityscapes.py [options]

Options:
    -h, --help                  Print this message
    --NoLabels=<int>            The number of labels [default: 19]
    --ListPath=<str>            Input image dir list file [default: data/list/]
    --summaryPath=<str>         Path to the Tensorboard summary [default: data/summary/]
    --GPUID=<int>               The GPU trying to work on[default: 0]
    
    --lr=<float>                Learning Rate [default: 0.000025]
    --iterSize=<int>            Num iters to accumulate gradients over [default: 5]
    --wtDecay=<float>           Weight decay during training [default: 0.0005]
    --epoch=<int>               Epoch Number [default: 60]    
    --pretrain=<str>            Path to the pretrain model of the net [default: data/snapshots/cityscapes_epoch_59.pth]
"""
# --gpu0=<int> GPU number[default: 0]

args = docopt(docstr, version='v0.1')
# cudnn.enabled = False

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


class Cityscapes(object):
    def __init__(self, args):
        # import most params
        self.batch_size = 1
        self.weight_decay = float(args['--wtDecay'])
        self.base_lr = float(args['--lr'])
        self.epoch_num = int(args['--epoch'])
        self.label_num = int(args['--NoLabels'])
        self.GPU = int(args['--GPUID'])
        self.model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))

        self.model.load_state_dict(torch.load(args['--pretrain']))
        self.model.float()
        self.model.eval()
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     self.model = nn.DataParallel(self.model)

        self.model.cuda(self.GPU)

        # data set
        self.datasets = dataset.create_dataset(args['--ListPath'])

        # optimization
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)  # use a Classification Cross-Entropy loss
        self.optimizer = optim.SGD(
            [{'params': get_1x_lr_params_NOscale(self.model), 'lr': self.base_lr},
             {'params': get_10x_lr_params(self.model), 'lr': 10 * self.base_lr}],
              lr=self.base_lr, momentum=0.9, weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, gamma=0.5, last_epoch=-1)
        self.optimizer.zero_grad()

        # data writer
        self.writer = SummaryWriter(log_dir=args['--summaryPath'])

    def train_one_epoch(self, epoch):
        log('==================stat to train===================')
        self.optimizer.zero_grad()
        hist = np.zeros((self.label_num, self.label_num))
        # prepare data loader
        train_loader = dataset.DataLoader(self.datasets['train'],
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=8)
        total_batch_num = len(self.datasets['train'])/self.batch_size

        # train through dataset
        for batch_idx, batch in enumerate(train_loader):

            image, label = batch['image'], batch['label']
            image, label = image.cuda(self.GPU), label.cuda(self.GPU)

            out = self.model(image)[0]
            out = nn.UpsamplingBilinear2d(size=image.shape[-2:])(out)  # N C H W
            loss = self.criterion(out, label)
            iter_size = int(args['--iterSize'])
            # for i in range(len(out) - 1):
            #     loss += loss_calc(out[i + 1], label[i + 1], gpu0)
            loss = loss/iter_size
            loss.backward()

            if batch_idx % iter_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                # calculate IOU
                pred = torch.argmax(out, dim=1)
                hist = fast_hist(pred.cpu().data.numpy().flatten(),
                                 label.cpu().data.numpy().flatten(),
                                 self.label_num)
                ious = per_class_iu(hist) * 100
                mean_iou = mean_iu(ious)

                # output result
                print('epoch %d | %d/%d complete | loss: %.4f | IoU: %.4f' % (epoch, batch_idx + 1,
                                                                              total_batch_num,
                                                                              loss.item()*iter_size,
                                                                              mean_iou))
                summary_idx = int(total_batch_num * epoch + batch_idx)
                self.writer.add_scalar('training_loss', loss.item()*iter_size, summary_idx)
                self.writer.add_scalar('training_iou', mean_iou, summary_idx)

            if batch_idx % 100 == 0:
                id_map = logits2trainId(out[0])
                save_img = image[0].cpu().data.numpy().transpose(1, 2, 0)+70
                save_img = save_img.astype('uint8')
                summary_idx = int(total_batch_num * epoch + batch_idx)
                # self.writer.add_image('train_pred %d' % batch_idx, trainId2color(id_map), summary_idx)
                self.writer.add_image('train_image %d' % batch_idx, save_img, summary_idx)

    def eval_one_epoch(self, epoch):
        log('==================stat to val===================')
        self.optimizer.zero_grad()
        hist = np.zeros((self.label_num, self.label_num))

        # prepare data loader
        train_loader = dataset.DataLoader(self.datasets['val'],
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=8)
        total_batch_num = len(self.datasets['val'])/self.batch_size
        loss_sum = 0
        # train through dataset
        for batch_idx, batch in enumerate(train_loader):
            image, label = batch['image'], batch['label']
            image, label = image.cuda(self.GPU), label.cuda(self.GPU)

            out = self.model(image)[0]
            out = nn.UpsamplingBilinear2d(size=image.shape[-2:])(out)
            loss = self.criterion(out, label)
            loss_sum += loss.item()

            # cumulate confusion matrix
            pred = torch.argmax(out, dim=1)
            hist += fast_hist(pred.cpu().data.numpy().flatten(),
                              label.cpu().data.numpy().flatten(),
                              self.label_num)

            # save sample result to tensor board
            if batch_idx % 100 == 0:
                id_map = logits2trainId(out[0])  # save one image
                summary_idx = int(total_batch_num * epoch + batch_idx)
                save_img = image[0].cpu().data.numpy().transpose(1, 2, 0) + 70
                save_img = save_img.astype('uint8')
                # self.writer.add_image('val_pred %d' % batch_idx, trainId2color(id_map), summary_idx)
                self.writer.add_image('val_image %d' % batch_idx, save_img, summary_idx)
        # output result
        avg_loss = loss_sum/total_batch_num
        ious = per_class_iu(hist) * 100
        mean_iou = mean_iu(ious)

        self.writer.add_scalar('val_loss', avg_loss, epoch)
        self.writer.add_scalar('val_iou', mean_iou, epoch)
        log("avg_loss is %.4f" % avg_loss)
        log('mean IoU: %.3f' % mean_iou)
        print(' '.join('{:.03f}'.format(i) for i in ious))

    def train(self):
        for epoch_idx in range(self.epoch_num):
            print(self.optimizer)
            self.scheduler.step()
            self.train_one_epoch(epoch_idx)
            self.eval_one_epoch(epoch_idx)
            torch.save(self.model.state_dict(), 'data/snapshots/cityscapes1_epoch_%d.pth' % epoch_idx)
        self.writer.close()


if __name__ == "__main__":
    arg = docopt(docstr, version='v0.1')
    print(arg)
    start = time.time()
    model = Cityscapes(arg)
    model.train()
    model.eval_one_epoch(0)
    print("Running Time = %s", time.time()-start)
