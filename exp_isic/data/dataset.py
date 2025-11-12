import sys
import os
import random
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset, random_split
from exp_isic.data import transforms as T
from torchvision.transforms import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from copy import deepcopy


class ISICDataset(Dataset):
    def __init__(self, image_path='', stage='train', image_size=256, is_augmentation=False, labeled=True, percentage=0.1):
        super(ISICDataset, self).__init__()
        self.image_path = image_path
        self.image_size = image_size
        self.stage = stage
        self.is_augmentation = is_augmentation
        if self.stage == 'train':
            with open(self.image_path + "/train.list", "r") as f1:
                sample_list = f1.readlines()
                sample_list = [item.replace("\n", "") for item in sample_list]
                if labeled:
                    self.sample_list = sample_list[:int(len(sample_list)*percentage)]
                else:
                    self.sample_list = sample_list[int(len(sample_list)*percentage):]
        elif self.stage == 'val':
            with open(self.image_path + "/val.list", "r") as f1:
                sample_list = f1.readlines()
                sample_list = [item.replace("\n", "") for item in sample_list]
                self.sample_list = sample_list
        else:
            with open(self.image_path + "/test.list", "r") as f1:
                sample_list = f1.readlines()
                sample_list = [item.replace("\n", "") for item in sample_list]
                self.sample_list = sample_list
        if self.is_augmentation:
            self.augmentation = self.augmentation_transform()
        self.post_transform = self.post_transform()
        self.label_transform = self.label_transform()
        self.pre_transform = self.pre_transform()

    def __getitem__(self, item):
        image = io.imread(os.path.join(self.image_path, 'images', self.sample_list[item] + '.png')).astype('uint8')
        label = io.imread(os.path.join(self.image_path, 'masks', self.sample_list[item] + '.png')).astype('bool').astype('uint8')  # 255 --> 1
        image = Image.fromarray(image).convert('RGB')
        label = Image.fromarray(label).convert('L')
        if self.stage == 'train':
            image, label = self.pre_transform(image, label)
            imageA1, imageA2 = deepcopy(image), deepcopy(image)
            imageA1, _ = self.augmentation(imageA1, label)
            imageA2, _ = self.augmentation(imageA1, label)
            image, label = self.post_transform(image), self.label_transform(label)
            imageA1 = self.post_transform(imageA1)
            imageA2 = self.post_transform(imageA2)
            label = torch.from_numpy(np.array(label)).unsqueeze(0)
            return image, label, imageA1, imageA2
        elif self.stage == 'test':
            image, label = self.post_transform(image), self.label_transform(label)
            label = torch.from_numpy(np.array(label)).unsqueeze(0)
        else:
            image, label = self.post_transform(image), self.label_transform(label)
            label = torch.from_numpy(np.array(label)).unsqueeze(0)
        return image, label

    def __len__(self):
        return len(self.sample_list)

    @staticmethod
    def augmentation_transform():
        return T.Compose([
            T.ColorJitter(0.5, 0.5, 0.5, 0.05),
            T.RandomPosterize(bits=5, p=0.2),
            T.RandomAutocontrast(p=0.2),
            T.RandomEqualize(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    def pre_transform(self):
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
        ])

    def post_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.BILINEAR),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
        ])

    def label_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.NEAREST)
        ])

