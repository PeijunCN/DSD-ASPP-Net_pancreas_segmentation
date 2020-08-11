import numpy as np
import os
import sys
import time
from datasets import utils

path1 = '/home/hupj82/OrganSegRSTN_PyTorch-master/DATA2NPY/labels/'
path2 = '/home/hupj82/Data/npy_twostage'
result_path = path2

data_path = '/home/hupj82/OrganSegRSTN_PyTorch-master/DATA2NPY/'
organ_ID = int(1)
thres = 2
thres_rg = 1

numberoftest = 82
DSC_F1 = np.zeros(numberoftest)
DSC_F1P = np.zeros(numberoftest)

result_name = 'fusion_overlap'
result_rg_name = 'F1P_'+ str(thres_rg)

result_directory = os.path.join(result_path, result_name, 'volumes')
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

result_directory_rg = os.path.join(result_path, result_name, result_rg_name,'volumes')
if not os.path.exists(result_directory_rg):
    os.makedirs(result_directory_rg)

result_file = os.path.join(result_path, result_name, result_rg_name, 'results.txt')
output = open(result_file, 'w')
output.close()
output = open(result_file, 'a+')
output.write('Fusing results of ' + result_name + ':\n')
output.close()

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

    volume_file_F1 = utils.volume_filename_fusion(result_directory, 'F1', n) 
    pred_F1 = np.load(volume_file_F1)['volume'].astype(np.uint8)
    DSC_F1[i], inter_sum, pred_sum, label_sum = utils.DSC_computation(label, pred_F1)
    print('    DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
        str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .')
    output = open(result_file, 'a+')
    output.write('    DSC_F1 = 2 * ' + str(inter_sum) + ' / (' + \
        str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC_F1[i]) + ' .\n')
    output.close()

    volume_file_F1P = utils.volume_filename_fusion(result_directory_rg, 'F1P', n)
    S = pred_F1
    if (S.sum() == 0):
        S = pred_F1

    if os.path.isfile(volume_file_F1P):
        pred_F1P = np.load(volume_file_F1P)['volume'].astype(np.uint8)
    else:
        pred_F1P = utils.post_processing(pred_F1, S, thres_rg, organ_ID)
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


    pred_F1 = None
    pred_F1P = None

output = open(result_file, 'a+')
print('Average DSC_F1 = ' + str(np.mean(DSC_F1)) + ' .')
output.write('Average DSC_F1 = ' + str(np.mean(DSC_F1)) + ' .\n')
print('Average DSC_F1P = ' + str(np.mean(DSC_F1P)) + ' .')
output.write('Average DSC_F1P = ' + str(np.mean(DSC_F1P)) + ' .\n')
output.close()
print('The fusion process is finished.')
