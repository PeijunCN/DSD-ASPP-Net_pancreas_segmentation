import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
import torchvision.transforms as standard_transforms
from datasets import utils
import math

#num_classes = 19
num_classes = 3
ignore_label = 255

#root = '/home/hup/hu/MedicalData/pancreas/pic2'
#root = '/home/hup/hu/MedicalData/pancreas/crop_pic_axial'
root = '/home/hup/hu/MedicalData/pancreas/prob_crop_pic_axial'
cv_folder = 'cv1_li st'
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
    def __init__(self, mode, data_path, current_fold, organ_number, low_range, high_range, slice_threshold, slice_thickness, \
                 organ_ID, plane, joint_transform=None, sliding_crop=None, transform=None, target_transform=None, \
                 color_transform=None):
        self.low_range = low_range
        self.high_range = high_range
        self.slice_thickness = slice_thickness
        self.organ_ID = organ_ID

        image_list = open(utils.training_set_filename(current_fold), 'r').read().splitlines()
        self.training_image_set = np.zeros((len(image_list)), dtype=np.int)
        for i in range(len(image_list)):
            s = image_list[i].split(' ')
            self.training_image_set[i] = int(s[0])

        slice_list = open(utils.list_training[plane], 'r').read().splitlines()
        self.slices = len(slice_list)
        self.image_ID = np.zeros((self.slices), dtype=np.int)
        self.slice_ID = np.zeros((self.slices), dtype=np.int)
        self.image_filename = ['' for l in range(self.slices)]
        self.label_filename = ['' for l in range(self.slices)]
        self.prob_filename = ['' for l in range(self.slices)]
        self.average = np.zeros((self.slices))
        self.pixels = np.zeros((self.slices), dtype=np.int)

        for l in range(self.slices):
            s = slice_list[l].split(' ')
            self.image_ID[l] = s[0]
            self.slice_ID[l] = s[1]
            self.image_filename[l] = s[2]  # important
            self.label_filename[l] = s[3]  # important
            ss = s[3].split('labels')
            sss = ss[1].split('.npy')
            self.prob_filename[l] = ss[0] + 'preds' + sss[0] + '.npy'
            self.pixels[l] = int(s[organ_ID * 5])  # sum of label
        if slice_threshold <= 1:  # 0.98
            pixels_index = sorted(range(self.slices), key=lambda l: self.pixels[l])
            last_index = int(math.floor((self.pixels > 0).sum() * slice_threshold))
            min_pixels = self.pixels[pixels_index[-last_index]]
        else:  # or set up directly
            min_pixels = slice_threshold
        #self.active_index = [l for l, p in enumerate(self.pixels)
        #                     if p >= min_pixels and self.image_ID[l] in self.training_image_set]  # true active

        self.active_index = [l for l, p in enumerate(self.pixels)
                             if self.image_ID[l] in self.training_image_set]  # true active


        if len(self.image_filename) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.color_transform = color_transform

    def __getitem__(self, index):
        self.index1 = self.active_index[index]
        self.index0 = self.index1 - 1
        if self.index1 == 0 or self.slice_ID[self.index0] != self.slice_ID[self.index1] - 1:
            self.index0 = self.index1
        self.index2 = self.index1 + 1
        if self.index1 == self.slices - 1 or self.slice_ID[self.index2] != self.slice_ID[self.index1] + 1:
            self.index2 = self.index1

        if self.mode == 'test':
            self.data, self.label = self.load_data()
            self.data = self.data * 255
            img = Image.fromarray(self.data.astype(np.uint8)).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            img_name = self.image_filename[self.index1]
            return img_name, img

        self.data, self.label, self.prob = self.load_data()
        self.data = self.data*255
        img = np.transpose(self.data, (1, 2, 0))
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
        #img.show()
        mask = np.transpose(self.label, (1, 2, 0))
        prob = np.transpose(self.prob, (1, 2, 0))
        if self.slice_thickness==1:
            mask = mask[:, :, 0]*255
            mask = Image.fromarray(mask).convert('P')
            prob = prob[:, :, 0]
            prob = Image.fromarray(prob).convert('P')
        else:
            mask = mask*255
            mask = Image.fromarray(mask.astype(np.uint8)).convert('RGB')
            prob = prob
            prob = Image.fromarray(prob.astype(np.uint8)).convert('RGB')
        #mask.show()

        if self.joint_transform is not None:
            img, mask, prob = self.joint_transform(img, mask, prob)

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
                #img = standard_transforms.ToTensor()(img)
            if self.target_transform is not None:
                #mask = self.target_transform(mask)
                mask = standard_transforms.ToTensor()(mask)
                prob = standard_transforms.ToTensor()(prob)

            return img, mask, prob

    def __len__(self):
        return len(self.active_index)

    def load_data(self):
        if self.slice_thickness == 1:
            image1 = np.load(self.image_filename[self.index1]).astype(np.float32)
            label1 = np.load(self.label_filename[self.index1])
            prob1 = np.load(self.prob_filename[self.index1])
            width = label1.shape[0]
            height = label1.shape[1]
            image = np.repeat(image1.reshape(1, width, height), 3, axis=0)
            label = label1.reshape(1, width, height)
            prob = prob1.reshape(1, width, height)
        elif self.slice_thickness == 3:
            image0 = np.load(self.image_filename[self.index0])
            width = image0.shape[0]
            height = image0.shape[1]
            image = np.zeros((3, width, height), dtype=np.float32)
            image[0, ...] = image0
            image[1, ...] = np.load(self.image_filename[self.index1])
            image[2, ...] = np.load(self.image_filename[self.index2])
            label = np.zeros((3, width, height), dtype=np.uint8)
            label[0, ...] = np.load(self.label_filename[self.index0])
            label[1, ...] = np.load(self.label_filename[self.index1])
            label[2, ...] = np.load(self.label_filename[self.index2])
            prob = np.zeros((3, width, height), dtype=np.uint8)
            prob[0, ...] = np.load(self.prob_filename[self.index0])
            prob[1, ...] = np.load(self.prob_filename[self.index1])
            prob[2, ...] = np.load(self.prob_filename[self.index2])
        np.minimum(np.maximum(image, self.low_range, image), self.high_range, image)
        image -= self.low_range
        image /= (self.high_range - self.low_range)
        label = utils.is_organ(label, self.organ_ID).astype(np.uint8)
        return image, label, prob

