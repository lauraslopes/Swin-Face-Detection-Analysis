# Swin Transformer for Face Detection

This repo contains the supported code and configuration files to reproduce face detection results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

To convert WIDER FACE annotations to Pascal VOC XML format, please refer to [WIDER FACE PASCAL VOC ANNOTATIONS](https://github.com/akofman/wider-face-pascal-voc-annotations)

## Updates

***30/06/2026*** Refactoring and organizing repository

## Results and Models

### Cascade R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | #params | FLOPs | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | ImageNet-1K | 1x | 48.1 | 41.7 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/cascade_mask_rcnn_swin_tiny_patch4_window7_1x.log.json)/[baidu](https://pan.baidu.com/s/1x4vnorYZfISr-d_VUSVQCA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.3/cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth)/[baidu](https://pan.baidu.com/s/1vFwbN1iamrtwnQSxMIW4BA) |
| Swin-T | ImageNet-1K | 3x | 50.4 | 43.7 | 86M | 745G | [config](configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1GW_ic617Ak_NpRayOqPSOA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_tiny_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1i-izBrODgQmMwTv6F6-x3A) |
| Swin-S | ImageNet-1K | 3x | 51.9 | 45.0 | 107M | 838G | [config](configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/17Vyufk85vyocxrBT1AbavQ) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_small_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1Sv9-gP1Qpl6SGOF6DBhUbw) |
| Swin-B | ImageNet-1K | 3x | 51.9 | 45.0 | 145M | 982G | [config](configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.log.json)/[baidu](https://pan.baidu.com/s/1UZAR39g-0kE_aGrINwfVHg) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.2/cascade_mask_rcnn_swin_base_patch4_window7.pth)/[baidu](https://pan.baidu.com/s/1tHoC9PMVnldQUAfcF6FT3A) |

**Notes**: 

- **Pre-trained models can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer)**.
- Access code for `baidu` is `swin`.

## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

> conda create -n swin-det python=3.7 -y
> conda activate swin-det
> conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 \
>    cudatoolkit=11.0 -c pytorch -c conda-forge -y
> pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
> git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git
> cd Swin-Transformer-Object-Detection
> pip install -r requirements/build.txt
> pip install -v -e .
> pip install Cython==0.29.24
> pip install numpy==1.21.6
> pip install matplotlib==3.4.3
> pip install pycocotools==2.0.3
> pip install six==1.16.0
> pip install terminaltables==3.1.0
> pip install scipy==1.7.3
> pip install Pillow==8.4.0
> pip install timm==0.4.12
> pip install yapf==0.31.0
> python -c "import torch; import mmcv; import mmdet; \
>    print(torch.__version__, mmcv.__version__, mmdet.__version__)"
>
>> In any config under configs/swin/, remove or comment out this block:
>> # fp16 settings — remove this if not using Apex
>> fp16 = None
>> optimizer_config = dict(
>>    type="DistOptimizerHook",
>>    update_interval=1,
>>    grad_clip=None,
>>    coalesce=True,
>>    bucket_size_mb=-1,
>>    use_fp16=True,
>> )
>> And replace it with the standard optimizer config:
>> optimizer_config = dict(grad_clip=None)

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Citing Swin Transformer
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```

## Other Links

> **Image Classification**: See [Swin Transformer for Image Classification](https://github.com/microsoft/Swin-Transformer).

> **Semantic Segmentation**: See [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation).

> **Self-Supervised Learning**: See [MoBY with Swin Transformer](https://github.com/SwinTransformer/Transformer-SSL).

> **Video Recognition**, See [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer).

