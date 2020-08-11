import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
import torchvision.transforms as standard_transforms
import cv2
#num_classes = 19
num_classes = 2
ignore_label = 255

#root = '/home/hup/hu/MedicalData/pancreas/pic2'
#root = '/home/hup/hu/MedicalData/pancreas/crop_pic_axial'
root = '/home/hup/hu/MedicalData/pancreas/prob_crop_pic_axial'
cv_folder = 'cv4_li st'
'''
color map
0=background, 1=pancreas
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'train_prob', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'pancreas_jpg')
        mask_path = os.path.join(root, 'pancreas_label_png')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, cv_folder, 'pancreas_train')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    elif mode == 'train_prob':
        img_path = os.path.join(root, '3pancreas_jpg')
        mask_path = os.path.join(root, '3pancreas_label_png')
        prob_path = os.path.join(root, '3pancreas_prob_jpg')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, cv_folder, 'pancreas_train')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'), os.path.join(prob_path, it + '.jpg'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'pancreas_jpg')
        mask_path = os.path.join(root, 'pancreas_label_png')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root,  cv_folder, 'pancreas_test')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'pancreas_jpg')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, cv_folder, 'pancreas_test')).readlines()]
        for it in data_list:
            items.append((img_path, it + '.jpg'))
    return items


class VOC(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None, color_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.color_transform =color_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('P')

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.color_transform is not None:
            img = self.color_transform(img)

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]

            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)

            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)

            return img, mask

    def __len__(self):
        return len(self.imgs)


class PANCREAS(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None,
                 color_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.color_transform = color_transform

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        if self.mode == 'train_prob':
            img_path, mask_path, prob_path = self.imgs[index]
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('P')
            prob = Image.open(prob_path).convert('RGB')

            if self.joint_transform is not None:
                img, mask, prob = self.joint_transform(img, mask, prob)

            if self.color_transform is not None:
                img = self.color_transform(img)

            if self.transform is not None:
                img = self.transform(img)

            prob = standard_transforms.ToTensor()(prob)

            if self.target_transform is not None:
                mask = self.target_transform(mask)

            return img, mask, prob
        else:
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('P')
            # img.show()
            # print(img_path)
            if self.joint_transform is not None:
                img, mask = self.joint_transform(img, mask)

            if self.color_transform is not None:
                img = self.color_transform(img)

            if self.sliding_crop is not None:
                img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
                if self.transform is not None:
                    img_slices = [self.transform(e) for e in img_slices]
                if self.target_transform is not None:
                    mask_slices = [self.target_transform(e) for e in mask_slices]

                img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)

                return img, mask, torch.LongTensor(slices_info)
            else:
                if self.transform is not None:
                    img = self.transform(img)
                if self.target_transform is not None:
                    mask = self.target_transform(mask)

                return img, mask

    def __len__(self):
        return len(self.imgs)
