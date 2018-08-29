import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import math


def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j+1)/2
    j = int(np.ceil((j+1)/2.0))
    j = (j+1)/2
    return j


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return sorted(img_list)


def chunker(seq, size):
    '''
    rtype: generator for batches
    '''
    return (seq[pos:pos+size] for pos in range(0,len(seq), size))


def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)

    return label_resized


def flip(I, flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[0] * scale), int(img_temp.shape[1] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def scale_gt(img_temp, scale):
    new_dims = (  int(img_temp.shape[1]*scale),  int(img_temp.shape[0]*scale)   )
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)


def loss_calc(out, label, gpu0):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    # label = Variable(label).cpu()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss()
    out = m(out)


    return criterion(out,label)


def lr_poly(base_lr, iter, max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def random_crop(img, gt, crop_size=(512,512)):
    '''
    idtype: img(h*w*c), gt (h*w), size(ht,wt)
    rtype: (ht*wt*c), (ht*wt)
    '''
    h, w, c = img.shape
    ht, wt = crop_size
    i, j = random.randint(0, h - ht), random.randint(0, w - wt)
    cropped_img = img[i:(i + ht), j:(j + wt), :]
    cropped_gt = gt[i:(i + ht), j:(j + wt)]

    return cropped_img, cropped_gt


def add_axis(image):
    return image.view(1, *image.shape)


def log(x):
    print('\033[0;31;2m' + x + '\033[0m')


def save_plot(x, y, dir):
    plt.figure()
    # plt.ylim((0, 50))
    plt.title("Training Loss")
    plt.plot(x, y)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(dir)  # Save the figure
    log('finish plotting')


def init_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print('pred shape', pred.shape, 'gt shape', gt.shape)
    assert (pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = int(19) - 1  # labels from 0,1, ... 20(for VOC)
    count = np.zeros((max_label + 1,))
    for j in range(max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return aiou


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) * 1.0 / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def mean_iu(per_class_iou):
    valid_ious = per_class_iou[~np.isnan(per_class_iou)]
    valid_class = len(valid_ious)  # np.sum(valid_ious!= 0)
    return np.sum(valid_ious)/valid_class


def dev_bw_frame(key, cur):
    return np.sum(key != cur) * 1.0 / (np.prod(cur.shape))

#
# def kernel_wrap_layer(kernel, low_fea):
#
#
#     n, c, h, w = low_fea.size()
#     _, k2, _ ,_ = kernel.size()  # channel # of kernel is the k^2
#     kernel_unf = torch.nn.functional.unfold(kernel, (1, 1))  # n * k2 * (h*w)
#     n_channel_seq = [kernel_unf] * c
#     kernel_unf_n_channel = torch.cat(n_channel_seq, dim=1)  # n * (k2*c) * (h*w)
#
#     k = int(math.sqrt(k2))
#     low_fea_unf = torch.nn.functional.unfold(low_fea, (k, k), padding=k//2)  # n * k2*c * (h*w)
#
#
#     out_unf = low_fea_unf * kernel_unf_n_channel
#     out = torch.nn.functional.fold(out_unf, (h, w), (k, k))
#
#     return out


class kernel_wrap_layer(nn.Module):
    def __int__(self, kernel, low_fea):
        super(kernel_wrap_layer, self).__init__()

    def forward(self, kernel, low_fea):
        """
        :param kernel: n * k2 * h * w
        :param low_fea: n * c * h * w
        :return:
        """
        n, c, h, w = low_fea.size()
        _, k2, _, _ = kernel.size()  # channel # of kernel is the k^2
        kernel_unf = torch.nn.functional.unfold(kernel, (1, 1))  # n * k2 * (h*w)
        n_channel_seq = [kernel_unf] * c
        kernel_unf_n_channel = torch.cat(n_channel_seq, dim=1)  # n * (k2*c) * (h*w)

        k = int(math.sqrt(k2))
        low_fea_unf = torch.nn.functional.unfold(low_fea, (k, k), padding=k // 2)  # n * k2*c * (h*w)

        out_unf = low_fea_unf * kernel_unf_n_channel
        out = torch.nn.functional.fold(out_unf, (h, w), (k, k), padding=k//2)

        return out

