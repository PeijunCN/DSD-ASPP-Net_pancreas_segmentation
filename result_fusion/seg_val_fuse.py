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
    result_path = '/home/hupj82/Data/npy_twostage/fusion_overlap/F1P_0.1'
    path2 = os.path.join(result_path, 'volumes')

    result_file = os.path.join(result_path, 'results_F1P.txt')
    output = open(result_file, 'w')
    output.close()

    diceval= np.zeros(82)
    DSC_F1 = np.zeros(82)
    for n in range(82):
        volumeID = '{:0>4}'.format(n + 1)
        filename1 = volumeID + '.npy'
        directory1 = os.path.join(path1, filename1)

        #filename2 = volumeID + '.npy'
        filename2 = 'F1P_'+str(n+1)+'.npz'
        directory2 = os.path.join(path2, filename2)

        data1 = np.load(directory1).astype(np.uint8)
        #data2 = np.load(directory2).astype(np.uint8)
        #data2 = (data2 >= 128).astype(float)
        data2 = np.load(directory2)['volume'].astype(np.uint8)
        
        i = n
        DSC_F1[i], inter_sum, pred_sum, label_sum = utils.DSC_computation(data1, data2)
    
        output = open(result_file, 'a+')
        output.write('  Testcase ' + str(i + 1) + ':  DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + \
                     str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .\n')
        output.close()

        print(DSC_F1[i])
    
    print('Average DSC = ' + str(np.mean(DSC_F1)) + ' .')
    output = open(result_file, 'a+')
    output.write('Average DSC = ' + str(np.mean(DSC_F1)) + ' .\n')
    output.close()
  

if __name__ == '__main__':
    main()




