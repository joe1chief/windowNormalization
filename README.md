# Window Normalization (WIN)

[![Paper](https://img.shields.io/badge/paper-arXiv:2207.03366-b31b1b.svg)](https://arxiv.org/abs/2207.03366)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper:

> **A simple normalization technique using window statistics to improve the out-of-distribution generalization on medical images**<br>
> Chengfeng Zhou, Jun Wang, Suncheng Xiang, Feng Liu, Hefeng Huang, Dahong Qian<br>
> *IEEE Transactions on Medical Imaging (TMI), 2024*

<p align="center">
  <img src="assets/WIN-WIN.jpg" width="750">
</p>

## 1. Introduction

Convolutional Neural Networks (CNNs) often struggle with out-of-distribution (OOD) data, a common challenge in real-world clinical applications where data scarcity and heterogeneity are prevalent. To address this, we introduce **Window Normalization (WIN)**, a simple yet effective alternative to traditional normalization layers like Batch Normalization.

WIN perturbs the normalizing statistics (mean and standard deviation) with local statistics computed on a randomly cropped *window* of the feature map. This acts as a powerful feature-level augmentation, regularizing the model and significantly improving its OOD generalization. Building on this, we also propose **WIN-WIN**, a self-distillation method that uses a consistency loss between the model's predictions in training mode (with WIN) and evaluation mode (standard instance normalization).

Our extensive experiments across 6 tasks and 24 datasets demonstrate the general applicability and effectiveness of WIN and WIN-WIN.

## 2. Usage

### 2.1. Installation

First, clone the repository and install the required dependencies:

```bash
git clone https://github.com/joe1chief/windowNormalization.git
cd windowNormalization
pip install -r requirements.txt
```

### 2.2. Integrating WIN into Your Model

To replace all `torch.nn.BatchNorm2d` layers in your model with our `WindowNorm2d` layer, use the provided helper function:

```python
import torchvision.models as models
from WIN import WindowNorm2d

# Instantiate your model
net = models.resnet18(weights=None)

# Convert all BatchNorm2d layers to WindowNorm2d
net = WindowNorm2d.convert_WIN_model(net)
```

The `convert_WIN_model` function and the `WindowNorm2d` layer itself offer several hyper-parameters to control the normalization behavior. See the docstrings in `WIN.py` for a detailed explanation.

### 2.3. Training on CIFAR

The `cifar.py` script provides a complete example for training a ResNet-18 on CIFAR-10 and CIFAR-100, with support for evaluating on the corruption benchmarks (CIFAR-C, CIFAR-C-Bar).

**Download CIFAR-C:**

Before training, download the corruption datasets:

```bash
# Create data directory
mkdir -p ./data/cifar

# Download and extract CIFAR-10-C
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar -xvf CIFAR-10-C.tar -C ./data/cifar/

# Download and extract CIFAR-100-C
wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
tar -xvf CIFAR-100-C.tar -C ./data/cifar/
```

**Run Training:**

- **Train with WIN on CIFAR-10:**
  ```bash
  python cifar.py --dataset cifar10 --data-path ./data --norm WIN
  ```

- **Train with WIN-WIN on CIFAR-100:**
  ```bash
  python cifar.py --dataset cifar100 --data-path ./data --norm WIN-WIN
  ```

Checkpoints and training logs will be saved to a timestamped directory inside `./snapshots`.

## 3. Results

Performance of ResNet-18 on CIFAR-10/100 and their corresponding corruption benchmarks (CIFAR-C/100-C). `mCE` denotes mean Corruption Error (lower is better).

| Normalization | CIFAR-10 Acc. (%) | CIFAR-10-C mCE (%) | CIFAR-100 Acc. (%) | CIFAR-100-C mCE (%) |
| :--- | :---: | :---: | :---: | :---: |
| BatchNorm | 94.0 ± 0.2 | 25.8 ± 0.3 | 74.8 ± 0.2 | 51.5 ± 0.7 |
| GroupNorm | 91.2 ± 1.2 | 23.6 ± 1.8 | 66.1 ± 0.9 | 55.5 ± 0.5 |
| InstanceNorm | **94.4 ± 0.1** | 18.4 ± 0.3 | 74.4 ± 0.3 | 48.7 ± 0.6 |
| **WIN (ours)** | 94.1 ± 0.1 | **18.3 ± 0.3** | **74.7 ± 0.2** | **46.7 ± 0.4** |

## 4. Pre-trained Models

| Dataset | Model | Download |
| :--- | :--- | :---: |
| CIFAR-10 | ResNet-18 + WIN (180 epochs) | [`link`](https://drive.google.com/file/d/1p0pfo4rafBSfIl9pl39ylbnA6XYSnZ-o/view?usp=share_link) |
| CIFAR-100 | ResNet-18 + WIN (200 epochs) | [`link`](https://drive.google.com/file/d/1eTTVJyYPP41Lh_1lx9QcFHt1AFAaX99_/view?usp=share_link) |

## 5. Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{zhou2024simple,
  title={A simple normalization technique using window statistics to improve the out-of-distribution generalization on medical images},
  author={Zhou, Chengfeng and Wang, Jun and Xiang, Suncheng and Liu, Feng and Huang, Hefeng and Qian, Dahong},
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  publisher={IEEE}
}
```

## Contact

For any questions or discussions, please feel free to contact **Chengfeng Zhou** at `joe1chief1993@gmail.com`.
