# DA-RTDETR: Multi-Level Domain Adaptation for Real-Time Detection Transformers

**Yuan Ze University — Intelligent Computing Lab (IC Lab)**

> Muhammad Waqar   
> Advisor: Dr. Qazi Mazhar Ul Haq  
> 

---

## Overview

DA-RTDETR is a multi-level unsupervised domain adaptation framework built on RT-DETR-X with a ResNet-101 backbone. It aligns source and target domains at three complementary levels:

- **Pixel-level**: Domain-adversarial branch on backbone features (GRL + domain classifier)
- **Instance-level**: Domain-adversarial branch on encoder/query features (GRL + domain classifier)
- **CORAL alignment**: Correlation-alignment loss on decoder category-level tokens (second-order statistics matching)

## Results

| Benchmark | Metric | Baseline (Geng et al. 2026) | Ours |
|---|---|---|---|
| Cityscapes → Foggy Cityscapes | mAP@0.5 | 51.4% | **56.3%** |
| KITTI → Cityscapes | Car AP@0.5 | 50.6% | **51.6%** |
| Sim10K → Cityscapes | Car AP@0.5 | 51.4%* | **55.2%** |
| Cityscapes → BDD100K | mAP@0.5 | 40.6% | ≈32.6% (in progress) |

*Compared against Adaptive Teacher (Li et al., CVPR 2022)

## Installation

```bash
# Python 3.9+, PyTorch 2.0+, CUDA 11.8+
pip install -r requirement.txt
```

System used for experiments:
- OS: Windows 11
- Python: 3.9
- GPU: NVIDIA RTX 5070 Ti (16 GB)
- PyTorch: 2.0+
- Conda environment: confmix_car

## Dataset Preparation

Download and prepare datasets in COCO format:

- [Cityscapes](https://www.cityscapes-dataset.com/)
- [Foggy Cityscapes](https://www.cityscapes-dataset.com/)
- [KITTI](https://www.cvlibs.net/datasets/kitti/)
- [Sim10K](https://fcav.engin.umich.edu/projects/driving-in-the-matrix)
- [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/)

Update the config files with your dataset paths:

```yaml
# Example: configs/dataset/city2foggy_custom.yml
train_dataloader_source:
  dataset:
    img_folder: /path/to/cityscapes/train/images
    ann_file: /path/to/coco_json/cityscapes_train.json

train_dataloader_target:
  dataset:
    img_folder: /path/to/foggy/train/images
    ann_file: /path/to/coco_json/foggy_train.json
```

## Training

### Cityscapes → Foggy Cityscapes
```bash
python train.py -c configs/rtdetr/city2foggy_rtdetrx.yml --use_pixel_da --use_instance_da
```

### KITTI → Cityscapes
```bash
python train.py -c configs/rtdetr/kitti2city_rtdetrx.yml --use_pixel_da --use_instance_da
```

### Sim10K → Cityscapes
```bash
python train.py -c configs/rtdetr/sim10k2city_rtdetrx.yml --use_pixel_da --use_instance_da
```

### Cityscapes → BDD100K
```bash
python train.py -c configs/rtdetr/city2bdd_rtdetrx.yml --use_pixel_da --use_instance_da
```

### Resume training from checkpoint
```bash
python train.py -c configs/rtdetr/city2bdd_rtdetrx.yml --use_pixel_da --use_instance_da --resume output/rtdetrx_city2bdd/checkpoint.pth
```

## Project Structure

```
DA-RTDETR/
├── train.py                    # Main training script
├── configs/                    # Training configuration files
│   ├── dataset/               # Dataset configs
│   └── rtdetr/                # Model configs
├── src/
│   ├── da/                    # Domain adaptation modules
│   │   ├── da.py             # Pixel-level + Instance-level DA
│   │   └── grl.py            # Gradient Reversal Layer
│   ├── zoo/rtdetr/           # RT-DETR model
│   ├── nn/backbone/          # PResNet backbone
│   └── solver/               # Training engine
├── tools/                     # Utility scripts
├── benchmark/                 # Benchmarking tools
└── tsne_feature_alignment.py  # t-SNE visualization script
```

## Key Implementation: Domain Adaptation Modules

The domain adaptation code is in `src/da/da.py`:

- **Pixel-level DA**: Applied to backbone multi-scale features
- **Instance-level DA**: Applied to encoder output features  
- **CORAL loss**: Applied to decoder category-level tokens

The Gradient Reversal Layer is implemented in `src/da/grl.py`.

## Citation

If you use this code, please cite:

```bibtex
@article{geng2026dartdetr,
  title={DA-RTDETR: domain-adaptive RT-DETR with feature fusion and category-level constraints},
  author={Geng, Huantong and Wang, Yingrui and Liu, Zhenyu and Fang, Long and Fan, Zichen},
  journal={Complex \& Intelligent Systems},
  volume={12},
  pages={26},
  year={2026}
}
```

## Acknowledgment

This implementation builds upon:
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) by lyuwenyu
- [ConfMix](https://github.com/giuliorossolini/ConfMix) by Mattolin et al.
- [Deep CORAL](https://github.com/VisionLearningGroup/CORAL) by Sun & Saenko

Research conducted at the **Intelligent Computing Lab (IC Lab)**, Yuan Ze University, Taiwan.  
Lab website: https://sites.google.com/view/intelligentcomputinglab/members
