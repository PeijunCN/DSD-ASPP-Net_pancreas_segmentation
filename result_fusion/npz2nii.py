import numpy as np
import os
import nibabel


N = 82
W = 512
H = 512
islabel = 0


path1 = '/home/hupj82/Data/NIH/TCIA_pancreas_labels-02-05-2017'
path2 = '/home/hupj82/Data/npy_twostage/fusion_overlap/volumes' #prediction as .npz
savepath = '/home/hupj82/Data/prediction/geo_twostage/fusion_overlap/'
#path2 = '/home/hupj82/Data/npy_twostage/fusion_overlap/F1P_0.1/volumes' #prediction as .npz
#savepath = '/home/hupj82/Data/prediction/geo_twostage/fusion_overlap_f1p0.1/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

for n in range(82):
    volumeID = '{:0>4}'.format(n + 1)
    print('Processing File ' + volumeID)
    filename1 = 'label' + volumeID + '.nii.gz'

    directory1 = os.path.join(path1, filename1)
    filename2 = 'F1_' + str(n+1) + '.npz'

    file1 = os.path.join(path1, filename1)
    img = nibabel.load(file1)


    file2 = os.path.join(path2, filename2)
    
    data = np.load(file2)['volume'].astype(np.uint8)
    data = data.transpose(1, 0, 2)
    data = np.flip(data, 2)
    print('Data shape is ' + str(data.shape) + ' .')

    img_save = nibabel.Nifti1Image(data, img._affine)
    path_save = os.path.join(savepath, filename1)
    img_save.to_filename(path_save)
    print('File ' + volumeID + ' is saved in ' + file2 + ' .')
