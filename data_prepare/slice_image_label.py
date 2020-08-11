import numpy as np
import os
import time
import utils
#from utils import *


# data_path: path of .npy data
# original image path: data_path/images
# data_path/labels
data_path = '/home/hup/hu/MedicalData/pancreas/raw/DSD_DATA/'
organ_number=1
folds= 4
low_range=-100
high_range =240

image_path = os.path.join(data_path, 'images')
image_path_ = {}
for plane in ['X', 'Y', 'Z']:
	image_path_[plane] = os.path.join(data_path, 'images_' + plane)
	if not os.path.exists(image_path_[plane]):
		os.makedirs(image_path_[plane])
label_path = os.path.join(data_path, 'labels')
label_path_ = {}
for plane in ['X', 'Y', 'Z']:
	label_path_[plane] = os.path.join(data_path, 'labels_' + plane)
	if not os.path.exists(label_path_[plane]):
		os.makedirs(label_path_[plane])
list_path = os.path.join(data_path, 'lists')
if not os.path.exists(list_path):
	os.makedirs(list_path)
list_training = {}
for plane in ['X', 'Y', 'Z']:
	list_training[plane] = os.path.join(list_path, 'training_' + plane + '.txt')

image_list = []
image_filename = []
keyword = ''
for directory, _, file_ in os.walk(image_path):
	for filename in sorted(file_):
		if keyword in filename:
			image_list.append(os.path.join(directory, filename))
			image_filename.append(os.path.splitext(filename)[0])
label_list = []
label_filename = []
for directory, _, file_ in os.walk(label_path):
	for filename in sorted(file_):
		if keyword in filename:
			label_list.append(os.path.join(directory, filename))
			label_filename.append(os.path.splitext(filename)[0])
print('image_list', len(image_list))
print('label_list', len(label_list))
if len(image_list) != len(label_list):
	exit('Error: the number of labels and the number of images are not equal!')
total_samples = len(image_list)

for plane in ['X', 'Y', 'Z']:
	output = open(list_training[plane], 'w') # create txt
	output.close()
print('Initialization starts.')
#len(image_list)
for i in range(len(image_list)):
	start_time = time.time()
	print('Processing ' + str(i + 1) + ' out of ' + str(total_samples) + ' files.')
	image = np.load(image_list[i])
	label = np.load(label_list[i])
	print('  3D volume is loaded: ' + str(time.time() - start_time) + ' second(s) elapsed.')
	for plane in ['X', 'Y', 'Z']:
		if plane == 'X':
			slice_number = label.shape[0]
		elif plane == 'Y':
			slice_number = label.shape[1]
		elif plane == 'Z':
			slice_number = label.shape[2]
		print('  Processing data on ' + plane + ' plane (' + str(slice_number) + ' slices): ' + \
			 str(time.time() - start_time) + ' second(s) elapsed.')
		image_directory_ = os.path.join(image_path_[plane], image_filename[i])
		if not os.path.exists(image_directory_):
			os.makedirs(image_directory_)
		label_directory_ = os.path.join(label_path_[plane], label_filename[i])
		if not os.path.exists(label_directory_):
			os.makedirs(label_directory_)
		print('    Slicing data: ' + str(time.time() - start_time) + ' second(s) elapsed.')
		sum_ = np.zeros((slice_number, organ_number + 1), dtype = np.int)
		minA = np.zeros((slice_number, organ_number + 1), dtype = np.int)
		maxA = np.zeros((slice_number, organ_number + 1), dtype = np.int)
		minB = np.zeros((slice_number, organ_number + 1), dtype = np.int)
		maxB = np.zeros((slice_number, organ_number + 1), dtype = np.int)
		average = np.zeros((slice_number), dtype = np.float)
		for j in range(0, slice_number):
			image_filename_ = os.path.join( \
				image_path_[plane], image_filename[i], '{:0>4}'.format(j) + '.npy')
			label_filename_ = os.path.join( \
				label_path_[plane], label_filename[i], '{:0>4}'.format(j) + '.npy')
			if plane == 'X':
				image_ = image[j, :, :]
				label_ = label[j, :, :]
			elif plane == 'Y':
				image_ = image[:, j, :]
				label_ = label[:, j, :]
			elif plane == 'Z':
				image_ = image[:, :, j]
				label_ = label[:, :, j]
			if not os.path.isfile(image_filename_) or not os.path.isfile(label_filename_):
				np.save(image_filename_, image_) # main function, no truncate
				np.save(label_filename_, label_)
			np.minimum(np.maximum(image_, low_range, image_), high_range, image_)
			
			average[j] = float(image_.sum()) / (image_.shape[0] * image_.shape[1])
			for o in range(1, organ_number + 1):
				sum_[j, o] = (utils.is_organ(label_, o)).sum()
				arr = np.nonzero(utils.is_organ(label_, o))
				minA[j, o] = 0 if not len(arr[0]) else min(arr[0]) # [A*B] min/max nonzero
				maxA[j, o] = 0 if not len(arr[0]) else max(arr[0])
				minB[j, o] = 0 if not len(arr[1]) else min(arr[1])
				maxB[j, o] = 0 if not len(arr[1]) else max(arr[1])
		print('    Writing training lists: ' + str(time.time() - start_time) + ' second(s) elapsed.')
		output = open(list_training[plane], 'a+')
		for j in range(0, slice_number):
			image_filename_ = os.path.join( \
				image_path_[plane], image_filename[i], '{:0>4}'.format(j) + '.npy')
			label_filename_ = os.path.join( \
				label_path_[plane], label_filename[i], '{:0>4}'.format(j) + '.npy')
			output.write(str(i) + ' ' + str(j))
			output.write(' ' + image_filename_ + ' ' + label_filename_)
			output.write(' ' + str(average[j]))
			for o in range(1, organ_number + 1):
				output.write(' ' + str(sum_[j, o]) + ' ' + str(minA[j, o]) + \
					' ' + str(maxA[j, o]) + ' ' + str(minB[j, o]) + ' ' + str(maxB[j, o]))
			output.write('\n')
		output.close()
		print('  ' + plane + ' plane is done: ' + \
			str(time.time() - start_time) + ' second(s) elapsed.')
	print('Processed ' + str(i + 1) + ' out of ' + str(total_samples) + ' files: ' + \
		str(time.time() - start_time) + ' second(s) elapsed.')


print('Writing training image list.')
for f in range(folds):
	training_set_filename = os.path.join(list_path, 'training_' + 'FD' + str(f) + '.txt')
	list_training_ = training_set_filename
	output = open(list_training_, 'w')
	for i in range(total_samples):
		if utils.in_training_set(total_samples, i, folds, f):
			output.write(str(i) + ' ' + image_list[i] + ' ' + label_list[i] + '\n')
	output.close()
print('Writing testing image list.')
for f in range(folds):
	testing_set_filename = os.path.join(list_path, 'testing_' + 'FD' + str(f) + '.txt')
	list_testing_ = testing_set_filename
	output = open(list_testing_, 'w')
	for i in range(total_samples):
		if not utils.in_training_set(total_samples, i, folds, f):
			output.write(str(i) + ' ' + image_list[i] + ' ' + label_list[i] + '\n')
	output.close()
print('Initialization is done.')

