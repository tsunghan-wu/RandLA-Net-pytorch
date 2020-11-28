# RandLA-Net-pytorch

This repository contains the implementation of **RandLA-Net** (CVPR 2020 Oral) in PyTorch.

## Environment Setup

0. Click [this webpage](https://pytorch.org/get-started/locally/) and use conda to install pytorch>=1.4 (Be aware of the cuda version when installation)

1. Install python packages

```
pip install -r requirements.txt
```

2. Compile C++ Wrappers

```
bash compile_op.sh
```

## Prepare Data

Download the [Semantic KITTI dataset](http://semantic-kitti.org/dataset.html#download), and preprocess the data:

```
python utils/data_prepare/data_prepare_semantickitti.py
```
Note: 
- Please change the dataset path in the `data_prepare_semantickitti.py` with your own path.
- Data preprocessing code will **convert the label to 0-19 index**

## Training & Testing

1. Training

```bash
python3 train_SemanticKITTI.py <args>
```

The training script will create `runs/` directory, which stores checkpoints & training history (tnsorboard event)

2. Testing

```bash
python3 test_SemanticKITTI.py <args>
```
**Note: if `--index_to_label` are set, output predictions will be ".label" files (label figure) which can be visualized; Otherwise, they will be ".npy" (0-19 index) files which is used to evaluated afterward.**

## Visualization & Evaluation

1. Visualization

```
python3 visualize_SemanticKITTI.py <args>
```

2. Evaluation

- Example Evaluation code
```
python3 evaluate_semantics.py --dataset /work/patrickwu2/PCL_Seg_data/sequences_0.06/ --predictions runs/supervised/predictions/ --sequences 8
```

## Acknowledgement

- Modify from [RandLA-Net PyTorch](https://github.com/qiqihaer/RandLA-Net-pytorch)
