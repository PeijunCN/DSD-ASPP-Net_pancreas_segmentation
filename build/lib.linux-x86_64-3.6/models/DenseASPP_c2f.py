import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torch.nn import BatchNorm2d as bn
import random
import numpy as np
import matplotlib.pyplot as plt
#import inspect
#from gpu_mem_track import MemTracker
#device =torch.device('cuda:0')
#frame = inspect.currentframe()
#gpu_tracker = MemTracker(frame)

class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """
    def __init__(self, model_cfg, n_class=2, output_stride=8):
        super(DenseASPP, self).__init__()

        bn_size = model_cfg['bn_size']
        drop_rate = model_cfg['drop_rate']
        growth_rate = model_cfg['growth_rate']
        num_init_features = model_cfg['num_init_features']
        block_config = model_cfg['block_config']

        dropout0 = model_cfg['dropout0']
        dropout1 = model_cfg['dropout1']
        d_feature0 = model_cfg['d_feature0']
        d_feature1 = model_cfg['d_feature1']

        feature_size = int(output_stride / 8)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', bn(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        # block1*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 1, block)
        num_features = num_features + block_config[0] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 1, trans)
        num_features = num_features // 2

        # block2*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 2, block)
        num_features = num_features + block_config[1] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=feature_size)
        self.features.add_module('transition%d' % 2, trans)
        num_features = num_features // 2

        # block3*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(2 / feature_size))
        self.features.add_module('denseblock%d' % 3, block)
        num_features = num_features + block_config[2] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 3, trans)
        num_features = num_features // 2

        # block4*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(4 / feature_size))
        self.features.add_module('denseblock%d' % 4, block)
        num_features = num_features + block_config[3] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 4, trans)
        num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', bn(num_features))
        if feature_size > 1:
            self.features.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))

        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)
        num_features = num_features + 5 * d_feature1
        #num_features = num_features + 3 * d_feature1

        self.classification2 = nn.Sequential(
            nn.Dropout2d(p=dropout1),
            nn.Conv2d(in_channels=num_features, out_channels=n_class, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, _input):

        feature = self.features(_input)

        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        cls = self.classification2(feature)


        return cls

class SADDenseNet(nn.Module):
    def __init__(self, model_cfg, n_class=3, output_stride=8, crop_margin=20, crop_prob=0.5, \
                 crop_sample_batch=1, TEST=None):
        super(SADDenseNet, self).__init__()
        self.TEST = TEST
        self.margin = crop_margin
        self.prob = crop_prob
        self.batch = crop_sample_batch
        self.n_class = n_class
        self.model = DenseASPP(model_cfg, n_class, output_stride)

    def forward(self, image, label=None, score=None, mode='coarse'):
        #gpu_tracker.track()
        if self.TEST is None:
            if mode =='coarse':
                # Coarse-scaled
                cropped_image, crop_info = self.crop_coarse(image.cpu())
                h = cropped_image.cuda()
                h = self.model(h)
                h = self.uncrop(crop_info, h.cpu(), image.cpu())

                return h
            else:

                cropped_image, crop_info = self.crop(label.cpu(), image.cpu())
                h = cropped_image.cuda()
                h = self.model(h)
                h = self.uncrop(crop_info.cpu(), h.cpu(), image.cpu())

                return h

        elif self.TEST == 'C': # Coarse testing
            h, crop_info = self.crop_coarse(image)
            h = self.model(h)
            h = self.uncrop(crop_info, h, image)
            return h

        elif self.TEST == 'O': # Oracle testing

            h, crop_info = self.crop(label, image)
            h = self.model(h)
            h = self.uncrop(crop_info, h, image)
            return h

        elif self.TEST == 'F': # Fine testing
            h, crop_info = self.crop(score, image)
            h = self.model(h)
            h = self.uncrop(crop_info, h, image)
            return h

        else:
            raise ValueError("wrong value of mode, should be in [None, 'C', 'O', 'F']")


    def crop(self, prob_map, saliency_data):
        (N, C, W, H) = prob_map.shape

        binary_mask = (prob_map >= 0.5)  # torch.uint8
        del prob_map
        #if label is not None and binary_mask.sum().item() == 0:
        #    binary_mask = (label >= 0.5)

        if self.TEST is not None:
            self.left = self.margin
            self.right = self.margin
            self.top = self.margin
            self.bottom = self.margin
        else:
            self.update_margin()

        if binary_mask.sum().item() == 0:  # avoid this by pre-condition in TEST 'F'
            minA = 0
            maxA = W
            minB = 0
            maxB = H
            self.no_forward = True
        else:
            if N > 1:
                mask = torch.zeros(size=(N, C, W, H))
                for n in range(N):
                    cur_mask = binary_mask[n, :, :, :]
                    arr = torch.nonzero(cur_mask)
                    minA = arr[:, 1].min().item()
                    maxA = arr[:, 1].max().item()
                    minB = arr[:, 2].min().item()
                    maxB = arr[:, 2].max().item()
                    bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
                            int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
                    mask[n, :, bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1
                    del arr, cur_mask
                #saliency_data = saliency_data * mask.cuda()
                saliency_data = saliency_data * mask

                del mask

            arr = torch.nonzero(binary_mask)
            minA = arr[:, 2].min().item()
            maxA = arr[:, 2].max().item()
            minB = arr[:, 3].min().item()
            maxB = arr[:, 3].max().item()
            self.no_forward = False
            del arr

        bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
                int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
        if(bbox[1]-bbox[0])% 8:
            a, b =divmod(bbox[1]-bbox[0], 8)
            bbox[1] = bbox[1] + (8-b)//2
            bbox[0] = bbox[0] - (8-b-(8 - b) // 2)

        if(bbox[3]-bbox[2])% 8:
            a, b =divmod(bbox[3]-bbox[2], 8)
            bbox[3] = bbox[3] + (8-b)//2
            bbox[2] = bbox[2] - (8-b-(8 - b) // 2)

        cropped_image = saliency_data[:, :, bbox[0]: bbox[1], bbox[2]: bbox[3]]

        if self.no_forward == True and self.TEST == 'F':
            cropped_image = torch.zeros_like(cropped_image).cuda()

        #crop_image1=cropped_image[0].numpy().transpose(1, 2, 0)
        #crop_image2 = cropped_image[1].numpy().transpose(1, 2, 0)
        ##plt.figure(figsize=(15, 5))
        #pt.subplot(1, 2, 1)
        #plt.imshow(crop_image1, cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(crop_image2, cmap='gray')
        #plt.show()

        crop_info = np.zeros((1, 4), dtype=np.int16)
        crop_info[0] = bbox
        #crop_info = torch.from_numpy(crop_info).cuda()
        crop_info = torch.from_numpy(crop_info)

        del binary_mask
        return cropped_image, crop_info

    def crop_coarse(self, image):
        (N, C, W, H) = image.shape

        self.left = 32
        self.right = 32
        self.top = 32
        self.bottom = 32

        bbox = [int(self.left), int(W-self.right), \
                int(self.top), int(H-self.bottom)]
        cropped_image = image[:, :, bbox[0]: bbox[1], bbox[2]: bbox[3]]

        crop_info = np.zeros((1, 4), dtype=np.int16)
        crop_info[0] = bbox
        crop_info = torch.from_numpy(crop_info)

        del bbox
        return cropped_image, crop_info

    def update_margin(self):
        MAX_INT = 256
        if random.randint(0, MAX_INT - 1) >= MAX_INT * self.prob:
            self.left = self.margin
            self.right = self.margin
            self.top = self.margin
            self.bottom = self.margin
        else:
            a = np.zeros(self.batch * 4, dtype=np.uint8)
            for i in range(self.batch * 4):
                a[i] = random.randint(0, self.margin * 2)
            self.left = int(a[0: self.batch].sum() / self.batch)
            self.right = int(a[self.batch: self.batch * 2].sum() / self.batch)
            self.top = int(a[self.batch * 2: self.batch * 3].sum() / self.batch)
            self.bottom = int(a[self.batch * 3: self.batch * 4].sum() / self.batch)

    def uncrop(self, crop_info, cropped_image, image):
        uncropped_image = torch.ones_like(image)
        uncropped_image *= (-9999999)
        bbox = crop_info[0]
        uncropped_image[:, :, bbox[0].item(): bbox[1].item(), bbox[2].item(): bbox[3].item()] = cropped_image
        return uncropped_image

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm_1', bn(input_num, momentum=0.0003)),

        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm_2', bn(num1, momentum=0.0003)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation_rate=1):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', bn(num_input_features)),
        self.add_module('relu_1', nn.ReLU(inplace=True)),
        self.add_module('conv_1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm_2', bn(bn_size * growth_rate)),
        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module('conv_2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, dilation=dilation_rate, padding=dilation_rate, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation_rate=1):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, dilation_rate=dilation_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', bn(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=stride))


class DSCLoss(nn.Module):
    def __init__(self):
        super(DSCLoss, self).__init__()
        self.epsilon = 0.000001
        return

    def forward(self, pred, target): # soft mode. per item.
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)
        DSC = (2 * (pred * target).sum(1) + self.epsilon) / ((pred + target).sum(1) + self.epsilon)
        return 1 - DSC.sum() / float(batch_num)



if __name__ == "__main__":
    model = DenseASPP(2)
    print(model)
