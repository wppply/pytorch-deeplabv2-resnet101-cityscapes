import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import read_file
import torchvision.transforms.functional as F
import cv2
import vis
import os
import numpy as np
import random

def calc_dataset_stats(dataloader):
    stack = torch.cat([sample['image'] for sample in dataloader], dim=0)
    img_mean = [np.mean(stack[:, i, :, :].numpy()) for i in range(3)]
    img_std = [np.std(stack[:, i, :, :].numpy()) for i in range(3)]
    return img_mean, img_std


def create_dataset(list_dir):
    """
    Create datasets for training, testing and validating
    :return datasets: a python dictionary includes three datasets
                        datasets[
    """
    phase = ['train', 'val']
    norm = Normalize(mean=[71.17702, 81.81791, 72.96148],
                     std=[1, 1, 1])
    # for 1080 Ti, max resolution allowed is 720
    transform = {'train': transforms.Compose([
                                              RandomScale(0.5, 1.5),
                                              RandomCrop(713),
                                              RandomHorizontalFlip(),
                                              ToTensor(),
                                              norm
                                              ]),

                 'val': transforms.Compose([
                                            RandomCrop(713),
                                            ToTensor(),
                                            norm
                                            ])
                 }

    datasets = {p: Cityscapes(list_dir, mode=p, transforms=transform[p]) for p in phase}

    return datasets


class Cityscapes(Dataset):
    def __init__(self, list_dir, mode='train', transforms=None):
        """
        Create Dataset subclass on cityscapes dataset
        :param dataset_dir: the path to dataset root, eg. '/media/ubuntu/disk/cityscapes'
        :param mode: phase, 'train', 'test' or 'eval'
        :param transforms: transformation
        """

        self.list_dir = list_dir
        self.transforms = transforms

        # create image and label list
        self.image_list = []
        self.label_list = []
        if mode == 'train':
            self.image_list = read_file(os.path.join(list_dir, 'cityscapes_train.txt'))
            self.label_list = read_file(os.path.join(list_dir, 'cityscapes_train_gt.txt'))
        elif mode == 'val':
            self.image_list = read_file(os.path.join(list_dir, 'cityscapes_val.txt'))
            self.label_list = read_file(os.path.join(list_dir, 'cityscapes_val_gt.txt'))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Overrides default method
        tips: 3 channels of label image are the same
        """
        image = cv2.imread(self.image_list[index])
        label = cv2.imread(self.label_list[index], 0)  # label.size (1024, 2048, 3)
        # image_name = self.image_list[index]
        # label_name = self.label_list[index]

        sample = {'image': image, 'label': label}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class Normalize(object):
    """
    normalized given tensor to range [0,1]
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_size = min(h, w, self.output_size)

        top = random.randint(0, h - new_size)
        left = random.randint(0, w - new_size)

        image = image[top: top + new_size, left: left + new_size, :]

        label = label[top: top + new_size, left: left + new_size]

        sample['image'], sample['label'] = image, label

        return sample


class RandomScale(object):
    """
    Rescale the image in a sample to a random size
    """
    def __init__(self, low, high):
        self.scale_range = (low, high)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        scale = random.uniform(*self.scale_range)
        new_w, new_h = h * scale, w * scale

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h))
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        sample['image'], sample['label'] = image, label

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, output_stride=16):
        self.output_stride = output_stride

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)

        # reset label shape
        # w, h = label.shape[0]//self.output_stride, label.shape[1]//self.output_stride
        # label = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        # label[label == 255] = 19
        label = label.astype(np.int64)

        # normalize image
        # image /= 255

        sample['image'], sample['label'] = torch.from_numpy(image), torch.from_numpy(label)

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __call__(self, sample, p=0.5):
        image, label = sample['image'], sample['label']
        if np.random.uniform(0, 1) < p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        sample['image'], sample['label'] = image, label

        return sample


# if __name__ == '__main__':
#     list_dir = 'data/list'
#     dataset = create_dataset(list_dir)
#     loader = DataLoader(dataset['train'],
#                         batch_size=1,
#                         shuffle=True,
#                         num_workers=8)
#     # print(calc_dataset_stats(loader))
#     for idx, batch in enumerate(loader):
#
#         image = batch['image']
#         label = batch['label']
#         print(idx)
#         a = image[0].permute(1, 2, 0).detach().numpy() + 70
#         a = a.astype('uint8')
#         b = label[0].detach().numpy()
#         b[b==255] =0
#         vis.vis_segmentation(a, b)