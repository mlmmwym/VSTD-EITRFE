# Vietnamese Scene Text Detection via Edge Information and Text Region Feature Enhancement





## Install Manually 
```bash
conda create -n dbnet python=3.6
conda activate dbnet

conda install ipython pip

# python dependencies
pip install -r requirement.txt

# install PyTorch with cuda-10.1
# Note that you can change the cudatoolkit version to the version you want.
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch



## Requirements
* pytorch 1.4+
* torchvision 0.5+
* gcc 4.9+


## Data Preparation

Training data: prepare a text `train.txt` in the following format, use '\t' as a separator
```
./datasets/train/img/001.jpg	./datasets/train/gt/001.txt
```

Validation data: prepare a text `test.txt` in the following format, use '\t' as a separator
```
./datasets/test/img/001.jpg	./datasets/test/gt/001.txt
```
- Store images in the `img` folder
- Store groundtruth in the `gt` folder

The groundtruth can be `.txt` files, with the following format:
```
x1, y1, x2, y2, x3, y3, x4, y4, annotation
```


## Train
1. config the `dataset['train']['dataset'['data_path']'`,`dataset['validate']['dataset'['data_path']`in [config/icdar2015_resnet18_fpn_DBhead_polyLR.yaml](cconfig/icdar2015_resnet18_fpn_DBhead_polyLR.yaml)
* . single gpu train
```bash
bash singlel_gpu_train.sh
```
* . Multi-gpu training
```bash
bash multi_gpu_train.sh
```
## Test

[eval.py](tools/eval.py) is used to test model on test dataset

1. config `model_path` in [eval.sh](eval.sh)
2. use following script to test
```bash
bash eval.sh
```

## Predict 
[predict.py](tools/predict.py) Can be used to inference on all images in a folder
1. config `model_path`,`input_folder`,`output_folder` in [predict.sh](predict.sh)
2. use following script to predict
```
bash predict.sh
```
You can change the `model_path` in the `predict.sh` file to your model location. 



