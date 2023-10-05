# Project: Referring image Segmentation
## Motivation
It is necessary to perform various tasks in language as well as fixed tasks in human-robot interaction and image editing.
### Objective
RIS aims to generate segmentation mask for the target object referred by a given query expressions in natural language. 
## Dataset description (COCO)

The COCO object detection dataset is a large-scale image dataset designed to advance state-of-the-art techniques for the object detection task. Many researches used the
MS COCO 2017 instance segmentation dataset. This dataset includes 118K/5K images for train/val with 80 class instance labels. Typically, to conveniently evaluate models, the evaluation is carried out on a validation dataset.
## Technical limitations
existing methods does not capture the meaning of text queries and depend on specific words

## Baseline for understand the project
To help with understanding the project, i will introduce the most famous model used in the project.
https://github.com/yz93/LAVT-RIS

![Pipeline Image](pipeline.jpg)

### Requirements
PyTorch v1.7.1
Python 3.7.
CUDA 10.2

### Datasets
1. Follow instructions in the `./refer` directory to set up subdirectories
and download annotations.
This directory is a git clone (minus two data files that we do not need)
from the [refer](https://github.com/lichengunc/refer) public API.

2. Download images from [COCO](https://cocodataset.org/#download).
Please use the first downloading link *2014 Train images [83K/13GB]*, and extract
the downloaded `train_2014.zip` file to `./refer/data/images/mscoco/images`.

### The Initialization Weights for Training
1. Create the `./pretrained_weights` directory where we will be storing the weights.
```shell
mkdir ./pretrained_weights
```
2. Download [pre-trained classification weights of
the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth),
and put the `pth` file in `./pretrained_weights`.
These weights are needed for training to initialize the model.

### Trained Weights of LAVT for Testing
1. Create the `./checkpoints` directory where we will be storing the weights.
```shell
mkdir ./checkpoints
```
2. Download LAVT model weights (which are stored on Google Drive) using links below and put them in `./checkpoints`.
https://drive.google.com/file/d/13D-OeEOijV8KTC3BkFP-gOJymc6DLwVT/view?usp=sharing)

## Training command
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py --model lavt --dataset refcoco --model_id refcoco --batch-size 3 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco/output

## Testing command
python test.py --model lavt --swin_type base --dataset refcoco --split val --resume ./checkpoints/refcoco.pth --workers 4 --ddp_trained_weights --window12 --img_size 480

