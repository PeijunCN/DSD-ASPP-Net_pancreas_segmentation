# DSD-ASPP-Net_pancreas_segmentation
This repository contains the DSD-ASPP-Net model (Distance-based Saliency-aware DenseASPP Network) for semantic segmentation implemented in PyTorch.


## Usage

### 1.  **Clone the repository:**<br />

```
git clone https://github.com/PeijunCN/DSD-ASPP-Net_pancreas_segmentation.git
```


### 2. **Download the NIH pancreas segmentation dataset and pretrained model:**<br/>
1. visit [Here](http://academictorrents.com/details/80ecfefcabede760cdbdf63e38986501f7becd49), download NIH pancreas segmentation datset
2. download pretrained DenseASPP model from [here](https://drive.google.com/file/d/1TmGJXB73Ep1YE8u227g8zLqZf6lEvmXS/view?usp=sharing), and put it under ./PyTorch_Pretrained/DenseASPP/
### 3. Data Preparation
1. convert images and labels from .nii.gz to .npy format
2. set the data path as *data_path*, put images and labels to '*data_path*/images' and '*data_path*/labels', respectively. 
3. run /data_prepare/init_dataset-medical.py

### 4. Requirement
1. PyTorch 1.4.0
2. TensorBoard for PyTorch. [Here](https://github.com/lanpa/tensorboard-pytorch)  to install
3. Some other libraries (find what you miss when running the code :-P)
4. install the GeodistTK [Here](https://github.com/taigw/GeodisTK), run
```
    python setup.py build
    python setup.py install 
```
### 5. training
1. coarse-scaled DenseASPP model training:
```
python train_pancreas_c2f200_coarse.py
```
2. fine-scaled DSD-ASPP-Net model training:
```
python train_pancreas_c2f200_saliency.py
```
#### 6. testing
1. coarse-scaled model testing:
```
python test_organ_coarse_batch.py
```
2. fine-scaled model testing:
```
python test_organ_fine_batch.py
``` 
