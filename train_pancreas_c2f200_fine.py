import datetime
import os
import torch

import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from datasets import pancreas_nii
import utils.joint_transforms_prob as joint_transforms

from utils import check_mkdir
from torch.nn import BCEWithLogitsLoss
import utils.transforms as extended_transforms

from cfgs.DenseASPP161 import Model_CFG
from models.DenseASPP_c2f200_3_6_12 import SADDenseNet
from collections import OrderedDict


ckpt_path = 'ckpt'
exp_name = 'model_fine_c2f200_3_6_12'

writer = SummaryWriter(os.path.join(ckpt_path, exp_name))

args = {
    'epoch_num': 30,
    'lr': 3*1e-5,
    'weight_decay': 1e-5,
    'momentum': 0.95,
    'lr_patience': 100,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 20,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.1,  # randomly sample some validation results to display
    'batchsize': 1,
    'slice_thickness': 3
}


def main(train_args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    data_path = '/home/hup/hu/MedicalData/pancreas/raw/DSD_DATA/'
    current_fold = '3'
    organ_number = int(1)
    low_range = int(-100)
    high_range = int(240)
    slice_threshold = float(0.98)
    slice_thickness = int(3)
    organ_ID = int(1)
    GPU_ID = int(0)
    plane = 'Z'
    mean_std = ([0.290101, 0.328081, 0.286964], [0.182954, 0.186566, 0.184475])
    if slice_thickness == 1:
        num_classes = 3
    else:
        num_classes = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    net = SADDenseNet(Model_CFG, n_class=num_classes, output_stride=8, TEST=None)
    model_path = './PyTorch_Pretrained/DenseASPP/denseASPP161_795_2.pkl'
    #model_path = './ckpt/model_fine_c2f200_3_6_12/FD0:Z3_1_20.pth'
    is_local = False
    curr_epoch = 1
    weight = torch.load(model_path, map_location=lambda storage, loc: storage)

    if is_local:
        net.load_state_dict(weight)
    else:
        model_dict = net.state_dict()
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            if k !='module.classification.1.weight'and k!='module.classification.1.bias':
                name = 'model.'+k[7:]  # remove `module.`
                ss = name.split('.')
                if ss[1] != 'ASPP_24' and ss[1] != 'ASPP_18':
                    new_state_dict[name] = v
        model_dict.update(new_state_dict)
        net.load_state_dict(model_dict)

    for k, param in net.named_parameters():
        print(k)

    net.train()
    net.cuda()

    train_joint_transform = joint_transforms.Compose([joint_transforms.RandomRotate3(10)])

    color_transform = standard_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)

    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()


    train_set = pancreas_nii.pancreas('train', data_path=data_path, current_fold=int(current_fold), organ_number=organ_number, \
        low_range=low_range, high_range=high_range, slice_threshold=slice_threshold, slice_thickness=slice_thickness, \
		organ_ID=organ_ID, plane=plane, joint_transform=train_joint_transform, transform=input_transform, \
                                 target_transform=target_transform, color_transform=color_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=16, drop_last=True)


    criterion = BCEWithLogitsLoss(size_average=False).cuda()

    snapshot_path = os.path.join(ckpt_path, exp_name)
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], betas=(train_args['momentum'], 0.999))

    if len(train_args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + train_args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * train_args['lr']
        optimizer.param_groups[1]['lr'] = train_args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(train_args) + '\n\n')

    lr_initial = train_args['lr']
    total_epoch = train_args['epoch_num']
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        train(train_loader, net, criterion, optimizer, epoch, train_args)

        train_args['lr'] = lr_initial * pow(1 - epoch / (total_epoch*4), 0.9)
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * train_args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
        ], betas=(train_args['momentum'], 0.999))
        optimizer.step()

        # save weight for each epoch
        if (epoch+1)%2:
            snapshot_name = 'FD' + current_fold + ':' + plane + str(slice_thickness) + '_' + str(organ_ID) + '_' + str(epoch)
            torch.save(net.state_dict(), os.path.join(snapshot_path, snapshot_name + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(snapshot_path, 'opt_' + snapshot_name + '.pth'))

def train(train_loader, net, criterion, optimizer, epoch, train_args):

    train_loss = 0.0
    curr_iter = (epoch - 1) * len(train_loader)
    for i, (input, label, prob) in enumerate(train_loader):

        N = input.size(0)
        input = Variable(input)
        label = Variable(label)

        optimizer.zero_grad()
        output, label = net(input, label, mode='fine')
        

        if train_args['slice_thickness'] == 1:
            loss = criterion(output[:, 1], label[:, 0].float()) / N
        else:
            loss = criterion(output.cuda(), label.cuda().float()) / N
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        curr_iter += 1

        if (i+1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), loss
            ))

        del input, label, loss

    print('[epoch %d], [train loss %.5f]' % (epoch, train_loss / len(train_loader)))


if __name__ == '__main__':
    main(args)
