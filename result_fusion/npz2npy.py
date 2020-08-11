import os
import numpy as np

path2 = './npy_dense161/fine-20'
path1 = '/home/hupj82/DenseASPP_thickness3_master/ckpt/model_dense161_fine/FD3:Z3_1_22/volumes/'
for id in range(21):
    volumeID = '{:0>4}'.format(id + 62)
    filename1 = '22_'+str(id+1)+'.npz'
    #filename1 = 'F1P_'+str(id+1)+'.npz'
    file1 = os.path.join(path1, filename1)
    filename2 = volumeID +'.npy'
    file2 = os.path.join(path2, filename2)
    r = np.load(file1)
    

    np.save(file2, r['volume'])

