import time
from utils import *
from cfgs.DenseASPP161 import Model_CFG
from models.DenseASPP_c2f200_3_6_18 import SADDenseNet
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict

def DSC_computation(label, pred):
	pred_sum = pred.sum()
	label_sum = label.sum()
	inter_sum = np.logical_and(pred, label).sum()
	return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum



# data_path of test images
data_path = '/home/hupj82/OrganSegRSTN_PyTorch-master/DATA2NPY/'
list_path = os.path.join(data_path, 'lists')
# snapshot path
snapshot_path = './ckpt/model_coarse_c2f200_3_6_18/'
epoch_list = [16, 18, 20, 22, 24]
# path to save results
result_path = './ckpt/model_coarse_c2f200_3_6_18/'


current_fold = '3'
plane = 'Z'

GPU_ID = int(0)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
N_CLASS = 3
organ_number = int(1)
slice_thickness = int(3)
low_range = int(-100)
high_range = int(240)
slice_threshold = float(0.98)
organ_ID = int(1)

data_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.290101, 0.328081, 0.286964],
                                                           [0.182954, 0.186566, 0.184475])])
# snapshot list
snapshot = []
for i in range(len(epoch_list)):
    snapshot_name = 'FD' + str(current_fold) + ':' + plane + str(slice_thickness) + '_' + str(organ_number) + '_' + str(epoch_list[i])

    snapshot_directory = os.path.join(snapshot_path, snapshot_name + '.pth')
    print('Snapshot directory: ' + snapshot_directory + ' .')
    snapshot.append(snapshot_directory)
print(str(len(snapshot)) + ' snapshots are to be evaluated.')


volume_list_file = os.path.join(list_path, 'testing_' + 'FD' + str(current_fold) + '.txt')
volume_list = open(volume_list_file, 'r').read().splitlines()
while volume_list[len(volume_list) - 1] == '':
    volume_list.pop()
DSC = np.zeros((len(snapshot), len(volume_list)))

for t in range(len(snapshot)):
    result_name = 'FD' + str(current_fold) + ':' + plane + str(slice_thickness) + '_' + str(organ_number) + '_' + str(
        epoch_list[t])
    result_directory = os.path.join(result_path, result_name)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    result_directory = os.path.join(result_path, result_name, 'volumes')
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    result_file = os.path.join(result_path, result_name, 'results.txt')
    output = open(result_file, 'w')
    output.close()

    output = open(result_file, 'a+')
    output.write('Evaluating snapshot ' + str(epoch_list[t]) + ':\n')
    output.close()
    finished = True

    for i in range(len(volume_list)):
        volume_file = os.path.join(result_directory, str(epoch_list[t]) + '_' + str(i + 1) + '.npz')
        if not os.path.isfile(volume_file):
            finished = False
            break
    if not finished:
        net = SADDenseNet(Model_CFG, N_CLASS, output_stride=8, TEST='C')
        weight = torch.load(snapshot[t])
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
        net = net.cuda()



    for i in range(len(volume_list)):
        start_time = time.time()
        print('Testing ' + str(i + 1) + ' out of ' + str(len(volume_list)) + ' testcases, ' + \
              str(t + 1) + ' out of ' + str(len(snapshot)) + ' snapshots.')
        volume_file = os.path.join(result_directory, str(epoch_list[t]) + '_' + str(i + 1) + '.npz')
        s = volume_list[i].split(' ')
        label = np.load(s[2])

        label = (label == organ_ID).astype(np.uint8)

        if not os.path.isfile(volume_file):
            image = np.load(s[1]).astype(np.float32)
            np.minimum(np.maximum(image, low_range, image), high_range, image)
            image -= low_range
            image /= (high_range - low_range)
            print('  Data loading is finished: ' + \
                  str(time.time() - start_time) + ' second(s) elapsed.')
            pred = np.zeros(image.shape, dtype=np.float32)
            minR = 0
            if plane == 'X':
                maxR = image.shape[0]
                shape_ = (1, 3, image.shape[1], image.shape[2])
            elif plane == 'Y':
                maxR = image.shape[1]
                shape_ = (1, 3, image.shape[0], image.shape[2])
            elif plane == 'Z':
                maxR = image.shape[2]
                shape_ = (1, 3, image.shape[0], image.shape[1])
            for j in range(minR, maxR):
                print(j)
                if slice_thickness == 1:
                    sID = [j, j, j]
                elif slice_thickness == 3:
                    sID = [max(minR, j - 1), j, min(maxR - 1, j + 1)]
                if plane == 'X':
                    image_ = image[sID, :, :].transpose(1, 2, 0).astype(np.float32)
                elif plane == 'Y':
                    image_ = image[:, sID, :].transpose(0, 2, 1).astype(np.float32)
                elif plane == 'Z':
                    image_ = image[:, :, sID].astype(np.float32)

                image_ = image_ * 255
                test_img = Image.fromarray(image_.astype(np.uint8)).convert('RGB')
                test_img = Variable(data_transforms(test_img).unsqueeze(0).cuda(), volatile=True)
                pre = net.forward(test_img)
                pre = F.sigmoid(pre)
                pre = pre.data.cpu().numpy()

                if slice_thickness == 1:
                    out = pre[0, 1]
                    if plane == 'X':
                        pred[j, :, :] = out
                    elif plane == 'Y':
                        pred[:, j, :] = out
                    elif plane == 'Z':
                        pred[:, :, j] = out
                elif slice_thickness == 3:
                    out = pre.reshape(pre.shape[1], pre.shape[2], pre.shape[3])
                    if plane == 'X':
                        if j == minR:
                            pred[j: j + 2, :, :] += out[1: 3, :, :]
                        elif j == maxR - 1:
                            pred[j - 1: j + 1, :, :] += out[0: 2, :, :]
                        else:
                            pred[j - 1: j + 2, :, :] += out[...]
                    elif plane == 'Y':
                        if j == minR:
                            pred[:, j: j + 2, :] += out[1: 3, :, :].transpose(1, 0, 2)
                        elif j == maxR - 1:
                            pred[:, j - 1: j + 1, :] += out[0: 2, :, :].transpose(1, 0, 2)
                        else:
                            pred[:, j - 1: j + 2, :] += out[...].transpose(1, 0, 2)
                    elif plane == 'Z':
                        if j == minR:
                            pred[:, :, j: j + 2] += out[1: 3, :, :].transpose(1, 2, 0)
                        elif j == maxR - 1:
                            pred[:, :, j - 1: j + 1] += out[0: 2, :, :].transpose(1, 2, 0)
                        else:
                            pred[:, :, j - 1: j + 2] += out[...].transpose(1, 2, 0)
            if slice_thickness == 3:
                if plane == 'X':
                    pred[minR, :, :] /= 2
                    pred[minR + 1: maxR - 1, :, :] /= 3
                    pred[maxR - 1, :, :] /= 2
                elif plane == 'Y':
                    pred[:, minR, :] /= 2
                    pred[:, minR + 1: maxR - 1, :] /= 3
                    pred[:, maxR - 1, :] /= 2
                elif plane == 'Z':
                    pred[:, :, minR] /= 2
                    pred[:, :, minR + 1: maxR - 1] /= 3
                    pred[:, :, maxR - 1] /= 2
            print('  Testing is finished: ' + str(time.time() - start_time) + ' second(s) elapsed.')
            pred = np.around(pred * 255).astype(np.uint8)
            np.savez_compressed(volume_file, volume=pred)
            print('  Data saving is finished: ' + \
                  str(time.time() - start_time) + ' second(s) elapsed.')
            pred_temp = (pred >= 128)
        else:
            volume_data = np.load(volume_file)
            pred = volume_data['volume'].astype(np.uint8)
            print('  Testing result is loaded: ' + \
                  str(time.time() - start_time) + ' second(s) elapsed.')
            pred_temp = (pred >= 128)
        DSC[t, i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_temp)
        print('    DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + \
              ' + ' + str(label_sum) + ') = ' + str(DSC[t, i]) + ' .')
        output = open(result_file, 'a+')
        output.write('  Testcase ' + str(i + 1) + ': DSC = 2 * ' + str(inter_sum) + ' / (' + \
                     str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC[t, i]) + ' .\n')
        output.close()
        if pred_sum == 0 and label_sum == 0:
            DSC[t, i] = 0
        print('  DSC computation is finished: ' + \
              str(time.time() - start_time) + ' second(s) elapsed.')


    print('Snapshot ' + str(epoch_list[t]) + ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .')
    output = open(result_file, 'a+')
    output.write('Snapshot ' + str(epoch_list[t]) + \
		': average DSC = ' + str(np.mean(DSC[t, :])) + ' .\n')
    output.close()

print('The testing process is finished.')
for t in range(len(snapshot)):
	print('  Snapshot ' + str(epoch_list[t]) + ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .')
