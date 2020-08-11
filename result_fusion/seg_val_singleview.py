import numpy as np
import os
import nibabel
import torch
import matplotlib.pyplot as plt
from skimage import transform, data
from datasets import utils
import scipy.misc


def main():
    path1 = '/home/hupj82/OrganSegRSTN_PyTorch-master/DATA2NPY/labels'
    result_path = '/home/hupj82/Data/npy_coarse'
    path2 = os.path.join(result_path, 'coarse-20')
    thres = 0.5
    
    result_file = os.path.join(result_path, 'results_coarse_z_' + str(thres) + '.txt')
    output = open(result_file, 'w')
    output.close()
    
    numcase = 82

    diceval= np.zeros(numcase)
    DSC_F1 = np.zeros(numcase)
    for i in range(numcase):
        n = i
        volumeID = '{:0>4}'.format(n + 1)
        filename1 = volumeID + '.npy'
        directory1 = os.path.join(path1, filename1)

        filename2 = volumeID + '.npy'
        directory2 = os.path.join(path2, filename2)

        data1 = np.load(directory1).astype(np.uint8)
        data2 = np.load(directory2).astype(np.uint8)
        data2 = (data2 >= 256*thres).astype(np.uint8)
        
        i = n
        DSC_F1[i], inter_sum, pred_sum, label_sum = utils.DSC_computation(data1, data2)
    
        output = open(result_file, 'a+')
        output.write('  Testcase ' + str(i + 1) + ':  DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + \
                     str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .\n')
        output.close()

        print(DSC_F1[i])
    
    print('Average DSC = ' + str(np.mean(DSC_F1)) + ' .')
    output = open(result_file, 'a+')
    output.write('Average DSC = ' + str(np.mean(DSC_F1)) + ' .\n'+'std DSC = ' + str(np.std(DSC_F1)) + ' .\n')
    output.write('max DSC = ' + str(np.max(DSC_F1)) + ' .\n'+'min DSC = ' + str(np.min(DSC_F1)) + ' .\n')
    output.close()




def dicecal(pred, target):
    epsilon = 0
    (H, W, L)=pred.shape
    pred = pred.reshape([H*W*L, 1])
    target = target.reshape([H * W * L, 1])

    DSC = (2 * (pred * target).sum(0) + epsilon) / ((pred + target).sum(0) + epsilon)
    return DSC

if __name__ == '__main__':
    main()




