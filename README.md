# Project: Discovering objects and distinguish the instances of objects in images (Referring image Segmentation)
## Motivation
Image segmentation is one of the major tasks in the field of computer vision. Currently, image segmentation is used in a wide range of fields, such as medical image analysis, autonomous driving, security, satellites, and aerial photography, and it is expected to be used in even more fields in the future. Concomitant with increasing interest in the foreground and distinguishing identical instances within the same class, instance segmentation is actively being researched
To achieve generalized scene analysis, the ability to distinguish between instances of objects is required, and so concomitant with increasing interest in the foreground and distinguishing identical instances within the same class, instance segmentation is actively being researched.
### Objective
The goal of instance segmentation is to produce a pixel-wise segmentation map of the image, where each pixel is assigned to a specific object instance.
## Dataset description (COCO)

The COCO object detection dataset is a large-scale image dataset designed to advance state-of-the-art techniques for the object detection task. Many researches used the
MS COCO 2017 instance segmentation dataset. This dataset includes 118K/5K images for train/val with 80 class instance labels. Typically, to conveniently evaluate models, the evaluation is carried out on a validation dataset.


![Pipeline Image](pipeline.jpg)

## Baseline for understand the project
To help with understanding the project, i will introduce the most famous model(Mask R-CNN) used in the project.
https://github.com/yz93/LAVT-RIS

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

| [RefCOCO](https://drive.google.com/file/d/13D-OeEOijV8KTC3BkFP-gOJymc6DLwVT/view?usp=sharing) | [RefCOCO+](https://drive.google.com/file/d/1B8Q44ZWsc8Pva2xD_M-KFh7-LgzeH2-2/view?usp=sharing) | [G-Ref (UMD)](https://drive.google.com/file/d/1BjUnPVpALurkGl7RXXvQiAHhA-gQYKvK/view?usp=sharing) | [G-Ref (Google)](https://drive.google.com/file/d/1weiw5UjbPfo3tCBPfB8tu6xFXCUG16yS/view?usp=sharing) |
|---|---|---|---|

3. Model weights and training logs of the new lavt_one implementation are below.

| RefCOCO | RefCOCO+ | G-Ref (UMD) | G-Ref (Google) |
|:-----:|:-----:|:-----:|:-----:|
|[log](https://drive.google.com/file/d/1YIojIHqe3bxxsWOltifa2U9jH67hPHLM/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1xFMEXr6AGU97Ypj1yr8oo00uObbeIQvJ/view?usp=sharing)|[log](https://drive.google.com/file/d/1Z34T4gEnWlvcSUQya7txOuM0zdLK7MRT/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1HS8ZnGaiPJr-OmoUn4-4LVnVtD_zHY6w/view?usp=sharing)|[log](https://drive.google.com/file/d/14VAgahngOV8NA6noLZCqDoqaUrlW14v8/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/14g8NzgZn6HzC6tP_bsQuWmh5LnOcovsE/view?usp=sharing)|[log](https://drive.google.com/file/d/1JBXfmlwemWSvs92Rky0TlHcVuuLpt4Da/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1IJeahFVLgKxu_BVmWacZs3oUzgTCeWcz/view?usp=sharing)|

* The Prec@K, overall IoU and mean IoU numbers in the training logs will differ
from the final results obtained by running `test.py`,
because only one out of multiple annotated expressions is
randomly selected and evaluated for each object during training.
But these numbers give a good idea about the test performance.
The two should be fairly close.


## Training command
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 train.py --model lavt --dataset refcoco --model_id refcoco --batch-size 3 --lr 0.00005 --wd 1e-2 --swin_type base --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco/output

## Testing command
python test.py --model lavt --swin_type base --dataset refcoco --split val --resume ./checkpoints/refcoco.pth --workers 4 --ddp_trained_weights --window12 --img_size 480

