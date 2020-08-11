import numpy as np
import os
import time
import utils
#from utils import *


# data_path: path of .npy data
# probability map path: data_path/probs

data_path = '/home/hup/hu/MedicalData/pancreas/raw/DSD_DATA/'
organ_number=1
folds= 4
low_range=-100
high_range =240

image_path = os.path.join(data_path, 'images')
image_path_ = {}
for plane in ['X', 'Y', 'Z']:
	image_path_[plane] = os.path.join(data_path, 'probs_' + plane)
	if not os.path.exists(image_path_[plane]):
		os.makedirs(image_path_[plane])

image_list = []
image_filename = []
keyword = ''
for directory, _, file_ in os.walk(image_path):
	for filename in sorted(file_):
		if keyword in filename:
			image_list.append(os.path.join(directory, filename))
			image_filename.append(os.path.splitext(filename)[0])

print('image_list', len(image_list))

total_samples = len(image_list)


for i in range(len(image_list)):
	start_time = time.time()
	print('Processing ' + str(i + 1) + ' out of ' + str(total_samples) + ' files.')
	image = np.load(image_list[i])
	print('  3D volume is loaded: ' + str(time.time() - start_time) + ' second(s) elapsed.')
	for plane in ['X', 'Y', 'Z']:
		if plane == 'X':
			slice_number = image.shape[0]
		elif plane == 'Y':
			slice_number = image.shape[1]
		elif plane == 'Z':
			slice_number = image.shape[2]
		print('  Processing data on ' + plane + ' plane (' + str(slice_number) + ' slices): ' + \
			 str(time.time() - start_time) + ' second(s) elapsed.')
		image_directory_ = os.path.join(image_path_[plane], image_filename[i])
		if not os.path.exists(image_directory_):
			os.makedirs(image_directory_)

		print('    Slicing data: ' + str(time.time() - start_time) + ' second(s) elapsed.')

		for j in range(0, slice_number):
			image_filename_ = os.path.join( \
				image_path_[plane], image_filename[i], '{:0>4}'.format(j) + '.npy')

			if plane == 'X':
				image_ = image[j, :, :]
			elif plane == 'Y':
				image_ = image[:, j, :]
			elif plane == 'Z':
				image_ = image[:, :, j]
			if not os.path.isfile(image_filename_):
				np.save(image_filename_, image_)


