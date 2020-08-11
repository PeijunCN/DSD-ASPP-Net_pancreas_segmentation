import numpy as np
import os
import sys
import time
from datasets import utils

path1 = '/home/hupj82/OrganSegRSTN_PyTorch-master/DATA2NPY/labels/'
path2 = '/home/hupj82/Data/npy_coarse'
result_path = path2

data_path = '/home/hupj82/OrganSegRSTN_PyTorch-master/DATA2NPY/'
organ_ID = int(1)
thres = 2

numberoftest = 82
DSC_X = np.zeros(numberoftest)
DSC_Y = np.zeros(numberoftest)
DSC_Z = np.zeros(numberoftest)
DSC_F1 = np.zeros(numberoftest)
DSC_F1P = np.zeros(numberoftest)

result_name = 'fusion_mj'

result_directory = os.path.join(result_path, result_name, 'volumes')
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_file = os.path.join(result_path, result_name, 'results.txt')
output = open(result_file, 'w')
output.close()
output = open(result_file, 'a+')
output.write('Fusing results of ' + result_name + ':\n')
output.close()

res_list = {}
res_list['X'] = os.path.join(path2,'npy_x')
res_list['Y'] = os.path.join(path2,'npy_y')
res_list['Z'] = os.path.join(path2,'npy_z')

for i in range(82):
    n = i 
    start_time = time.time()
    print('Testing ' + str(n + 1) + ' out of ' + str(82) + ' testcases.')
    output = open(result_file, 'a+')
    output.write('  Testcase ' + str(i + 1) + ':\n')
    output.close()
    
    volumeID = '{:0>4}'.format(n + 1)
    filename1 = volumeID + '.npy'
    s = os.path.join(path1, filename1)
    label = np.load(s)
    label = utils.is_organ(label, organ_ID).astype(np.uint8)

    for plane in ['X', 'Y', 'Z']:
        pred = np.zeros(label.shape, dtype = np.float32)
        volume_file_ = os.path.join(res_list[plane], volumeID + '.npy')
        pred = np.load(volume_file_)
        if plane == 'X':
            pred_ = (pred >= 256*0.5)
        if plane == 'Y':
            pred_ =  (pred >= 256*0.5)
        if plane == 'Z':
            pred_ =  (pred >= 256*0.5)
        DSC_, inter_sum, pred_sum, label_sum = utils.DSC_computation(label, pred_)
        print('    DSC_' + plane + ' = 2 * ' + str(inter_sum) + ' / (' + \
            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_) + ' .')
        output = open(result_file, 'a+')
        output.write('    DSC_' + plane + ' = 2 * ' + str(inter_sum) + ' / (' + \
            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_) + ' .\n')
        output.close()
        if pred_sum == 0 and label_sum == 0:
            DSC_ = 0
        pred = pred/255
        if plane == 'X':
            pred_X = pred_.astype(np.uint8)
            DSC_X[i] = DSC_
        elif plane == 'Y':
            pred_Y = pred_.astype(np.uint8)
            DSC_Y[i] = DSC_
        elif plane == 'Z':
            pred_Z = pred_.astype(np.uint8)
            DSC_Z[i] = DSC_

    volume_file_F1 = utils.volume_filename_fusion(result_directory, 'F1', n) 

    if not os.path.isfile(volume_file_F1):
        pred_total = pred_X + pred_Y + pred_Z
    if os.path.isfile(volume_file_F1):
        pred_F1 = np.load(volume_file_F1)['volume'].astype(np.uint8)
    else:
        pred_F1 = (pred_total >= thres).astype(np.uint8)
        #print(pred_total.max().item())
        np.savez_compressed(volume_file_F1, volume = pred_F1)
    DSC_F1[i], inter_sum, pred_sum, label_sum = utils.DSC_computation(label, pred_F1)
    print('    DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' \
        + str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .')
    output = open(result_file, 'a+')
    output.write('    DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + \
        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .\n')
    output.close()
    if pred_sum == 0 and label_sum == 0:
        DSC_F1[i] = 0

    

    volume_file_F1P = utils.volume_filename_fusion(result_directory, 'F1P', n)
    S = pred_F1
    if (S.sum() == 0):
        S = pred_F1

    if os.path.isfile(volume_file_F1P):
        pred_F1P = np.load(volume_file_F1P)['volume'].astype(np.uint8)
    else:
        pred_F1P = utils.post_processing(pred_F1, S, 0.5, organ_ID)
        np.savez_compressed(volume_file_F1P, volume = pred_F1P)
    DSC_F1P[i], inter_sum, pred_sum, label_sum = utils.DSC_computation(label, pred_F1P)
    print('    DSC_F1P = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
        str(label_sum) + ') = ' + str(DSC_F1P[i]) + ' .')
    output = open(result_file, 'a+')
    output.write('    DSC_F1P = 2 * ' + str(inter_sum) + ' / (' + \
        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F1P[i]) + ' .\n')
    output.close()
    if pred_sum == 0 and label_sum == 0:
        DSC_F1P[i] = 0

    
    pred_X = None
    pred_Y = None
    pred_Z = None
    pred_F1 = None
    pred_F1P = None

output = open(result_file, 'a+')
print('Average DSC_X = ' + str(np.mean(DSC_X)) + ' .')
output.write('Average DSC_X = ' + str(np.mean(DSC_X)) + ' .\n')
print('Average DSC_Y = ' + str(np.mean(DSC_Y)) + ' .')
output.write('Average DSC_Y = ' + str(np.mean(DSC_Y)) + ' .\n')
print('Average DSC_Z = ' + str(np.mean(DSC_Z)) + ' .')
output.write('Average DSC_Z = ' + str(np.mean(DSC_Z)) + ' .\n')
print('Average DSC_F1 = ' + str(np.mean(DSC_F1)) + ' .')
output.write('Average DSC_F1 = ' + str(np.mean(DSC_F1)) + ' .\n')
print('Average DSC_F1P = ' + str(np.mean(DSC_F1P)) + ' .')
output.write('Average DSC_F1P = ' + str(np.mean(DSC_F1P)) + ' .\n')
output.close()
print('The fusion process is finished.')
