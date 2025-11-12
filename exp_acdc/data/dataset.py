import sys
import torch
from torch.utils.data.dataset import Dataset
from exp_acdc.data import transforms as T
from torchvision.transforms import *
import numpy as np
import h5py
from PIL import Image
from copy import deepcopy



class ACDCDataset(Dataset):
    def __init__(self, image_path='', stage='train', image_size=256, is_augmentation=False, labeled=True, percentage=0.1):
        super(ACDCDataset, self).__init__()
        self.image_size = image_size
        self.sep = '\\' if sys.platform[:3] == 'win' else '/'
        self.stage = stage
        self.is_augmentation = is_augmentation
        self.image_path = image_path
        if self.stage == 'train':
            with open(self.image_path + "/train_patient_shuffle.list", "r") as f1:
                patient_list = f1.readlines()
            patient_list = [item.replace("\n", "") for item in patient_list]
            if labeled:
                patient_list = patient_list[:int(len(patient_list)*percentage)]
            else:
                patient_list = patient_list[int(len(patient_list)*percentage):]
            with open(self.image_path + "/train_slices.list", "r") as f1:
                train_slices_list = f1.readlines()
            train_slices_list = [item.replace("\n", "") for item in train_slices_list]
            self.sample_list = [x.split('.')[0] for z in patient_list for x in train_slices_list if x.startswith(z)]
        elif self.stage == 'val':
            with open(self.image_path + "/val_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        else:
            with open(self.image_path + "/test_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if self.is_augmentation:
            self.augmentation = self.augmentation_transform()
        self.post_transform = self.post_transform()
        self.label_transform = self.label_transform()
        self.pre_transform = self.pre_transform()

    def __getitem__(self, item):
        case = self.sample_list[item]
        h5f = h5py.File(self.image_path + "/data/slices/{}.h5".format(case), "r")
        image = h5f["image"][:] * 255.
        label = h5f["label"][:]
        image = Image.fromarray(image).convert('L')
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
            image = self.post_transform(image)
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
            T.ColorJitter(0.5, 0.5, 0.5, 0.25),
            T.RandomAutocontrast(p=0.2),
            T.RandomEqualize(p=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    
    def pre_transform(self):
        return T.Compose([
            # T.RandomScale([0.8, 1.2]),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=180),
        ])

    def post_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.BILINEAR),
            ToTensor(),
        ])

    def label_transform(self):
        return Compose([
            Resize([self.image_size, self.image_size], InterpolationMode.NEAREST)
        ])