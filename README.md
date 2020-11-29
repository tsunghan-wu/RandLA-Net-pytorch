# RandLA-Net-pytorch

This repository contains the implementation of [RandLA-Net (CVPR 2020 Oral)](https://arxiv.org/abs/1911.11236) in PyTorch.
- We only support SemanticKITTI dataset now. (Welcome everyone to develop together and raise PR)
- Our model is almost as good as the original implementation. (Validation set : Our 52.9% mIoU vs original 53.1%)
- We place our pretrain-model in [`pretrain_model/checkpoint.tar`](pretrain_model/checkpoint.tar) directory.

### Performance

> Results on Validation Set (seq 08)

- Compare with original implementation

| Model                      | mIoU  |
| -------------------------- | ----- |
| Original Tensorflow        | 0.531 |
| Our Pytorch Implementation | 0.529 |

- Per class mIoU

| mIoU | car  | bicycle | motorcycle | truck | other-vehicle | person | bicyclist | motorcyclist | road | parking | sidewalk | other-ground | building | fence | vegetation | trunk | terrain | pole | traffic-sign |
| ---- | ------- | ---------- | ----- | ------------- | ------ | --------- | ------------ | ---- | ------- | -------- | ------------ | -------- | ----- | ---------- | ----- | ------- | ---- | ------------ | ---- |
| 52.9 | 0.919 | 0.122 | 0.290 | 0.660 | 0.444 | 0.515 | 0.676 | 0.000 | 0.912 | 0.421 | 0.759 | 0.001 | 0.878 | 0.354 | 0.844 | 0.595 | 0.741 | 0.517 | 0.414 |

## A. Environment Setup

0. Click [this webpage](https://pytorch.org/get-started/locally/) and use conda to install pytorch>=1.4 (Be aware of the cuda version when installation)

1. Install python packages

```
pip install -r requirements.txt
```

2. Compile C++ Wrappers

```
bash compile_op.sh
```

## B. Prepare Data

Download the [Semantic KITTI dataset](http://semantic-kitti.org/dataset.html#download), and preprocess the data:

```
python data_prepare_semantickitti.py
```
Note: 
- Please change the dataset path in the `data_prepare_semantickitti.py` with your own path.
- Data preprocessing code will **convert the label to 0-19 index**

## C. Training & Testing

1. Training

```bash
python3 train_SemanticKITTI.py <args>
```

2. Testing

```bash
python3 test_SemanticKITTI.py <args>
```
**Note: if the flag `--index_to_label` is set, output predictions will be ".label" files (label figure) which can be visualized; Otherwise, they will be ".npy" (0-19 index) files which is used to evaluated afterward.**

## D. Visualization & Evaluation

1. Visualization

```bash
python3 visualize_SemanticKITTI.py <args>
```

2. Evaluation

- Example Evaluation code

```bash
python3 evaluate_SemanticKITTI.py --dataset /tmp2/tsunghan/PCL_Seg_data/sequences_0.06/ \
    --predictions runs/supervised/predictions/ --sequences 8
```

## Acknowledgement

- Original Tensorflow implementation [link](https://github.com/QingyongHu/RandLA-Net)
- Our network & config codes are modified from [RandLA-Net PyTorch](https://github.com/qiqihaer/RandLA-Net-pytorch)
- Our evaluation & visualization codes are modified from [SemanticKITTI API](https://github.com/PRBonn/semantic-kitti-api)
