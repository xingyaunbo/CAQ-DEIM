<h1 align="center">CAQ-DEIM</h1>

<p align="center">
  <strong>Complexity-Aware Quality-Guided End-to-End Pig Detection in Complex Pig-Barn Environments</strong>
</p>

<p align="center">
  <a href="https://github.com/xingyaunbo/CAQ-DEIM/blob/main/LICENSE">
    <img alt="license" src="https://img.shields.io/badge/License-Apache%202.0-blue">
  </a>
  <a href="https://github.com/xingyaunbo/CAQ-DEIM/issues">
    <img alt="issues" src="https://img.shields.io/github/issues/xingyaunbo/CAQ-DEIM">
  </a>
  <a href="https://github.com/xingyaunbo/CAQ-DEIM/stargazers">
    <img alt="stars" src="https://img.shields.io/github/stars/xingyaunbo/CAQ-DEIM">
  </a>
</p>

<p align="center">
CAQ-DEIM is an NMS-free end-to-end pig detector developed for dense, overlapping, and boundary-ambiguous group-housed pig-barn scenes.
</p>

Repository status: The manuscript is being prepared for submission.Author information, paper DOI, released checkpoints, and the permanent archive DOI will be added when available.

1. Introduction

Pig images collected in commercial group-housed barns often contain densely distributed targets, mutual occlusion, close physical contact, blurred boundaries, low visibility, nighttime infrared scenes, lens contamination, and motion blur. These conditions can cause boundary ambiguity, missed detections, and inaccurate bounding-box localization.

CAQ-DEIM is built on the DEIM end-to-end detection framework and improves the baseline from three complementary perspectives:

CA-DOS: Complexity-Aware Dense O2O Sampling;

BCFE: Boundary-Context Feature Enhancement;

HQ-MAL: High-Quality Matchability-Aware Loss.

The method focuses primarily on improving localization quality at stricter IoU thresholds while retaining the NMS-free DEIM inference pipeline.

2. Method

2.1 CA-DOS

CA-DOS computes an image-level complexity score using target density and inter-box overlap:

S_i = alpha * Norm(n_i) + beta * m_i + gamma * Norm(p_i)

where:

n_i is the number of ground-truth boxes;

m_i is the maximum pairwise IoU between ground-truth boxes;

p_i is the number of overlapping box pairs;

Norm(.) denotes min-max normalization.

Final settings:

Parameter

Value

alpha

0.35

beta

0.45

gamma

0.20

High-complexity threshold

0.45

Medium-complexity threshold

0.25

High-complexity sampling probability

0.70

Medium-complexity sampling probability

0.35

Overlap-pair IoU threshold

0.10

CA-DOS changes only the selection of Mosaic auxiliary images during training. It does not modify Hungarian matching, the model architecture, or the inference pipeline, and therefore adds no inference cost.

2.2 BCFE

BCFE is inserted after channel projection of HGNetv2 features and before the Hybrid Encoder. It is applied to the P3 and P4 feature paths.

The module contains four parallel branches:

local texture extraction;

directional structure modeling;

contextual perception;

spatial boundary gating.

The branch outputs are concatenated and fused, followed by a residual connection with a learnable scaling parameter initialized to 0.1.

2.3 HQ-MAL

The original MAL quality target is:

t_i = q_i^gamma

HQ-MAL selectively compensates high-quality matched predictions:

t_i_HQ = q_i^gamma + eta * (q_i - q_i^gamma),  if q_i >= tau
t_i_HQ = q_i^gamma,                            otherwise

Final settings:

Parameter

Value

Quality modulation factor gamma

1.5

High-quality threshold tau

0.55

Compensation coefficient eta

0.25

CAQ-DEIM does not explicitly model pig-to-pig occlusion relationships. Instead, its three components improve difficult-sample exposure, boundary-context representation, and confidence-localization consistency.

3. Results

3.1 PigDetect test set

Method

AP

AP75

Parameters (M)

GFLOPs

DEIM

0.752

0.853

3.723

15.092

CAQ-DEIM

0.775

0.884

4.688

17.965

Improvement

+0.023

+0.031

+0.965

+2.873

CAQ-DEIM improves AP by 2.3 percentage points and AP75 by 3.1 percentage points over the original DEIM baseline.

3.2 Ablation study

Method

CA-DOS

BCFE

HQ-MAL

AP

AP75

DEIM

-

-

-

0.752

0.853

DEIM + CA-DOS

✓

-

-

0.765

0.871

DEIM + BCFE

-

✓

-

0.763

0.875

DEIM + CA-DOS + BCFE

✓

✓

-

0.769

0.872

CAQ-DEIM

✓

✓

✓

0.775

0.884

3.3 Complexity-subset evaluation

Complexity

DEIM AP

CAQ-DEIM AP

Delta AP

DEIM AP75

CAQ-DEIM AP75

Delta AP75

Full

0.752

0.775

+0.023

0.853

0.884

+0.031

Low

0.776

0.794

+0.018

0.886

0.902

+0.016

Medium

0.741

0.770

+0.029

0.859

0.890

+0.031

High

0.738

0.758

+0.020

0.831

0.864

+0.033

3.4 External evaluation

Dataset

Images

DEIM AP

CAQ-DEIM AP

DEIM AP75

CAQ-DEIM AP75

External Dataset 1

276

0.782

0.808

0.854

0.859

External Dataset 2

400

0.768

0.791

0.868

0.882

The external evaluation sets were used exclusively for final testing and were not used for training, validation-based model selection, parameter tuning, or model-structure design.

4. Dataset

CAQ-DEIM is evaluated on the public PigDetect object-detection subset of the PigBench benchmark.

Split

Images

Bounding boxes

Pen environments

Train

2,431

33,197

30

Validation

250

3,544

30

Test

250

5,436

1 independent pen

Total

2,931

42,177

31

The training and validation sets contain images from 30 pen environments. The official test set comes from one independent pen environment that is absent from both training and validation.

Dataset DOI:

https://doi.org/10.25625/I6UYE9

The dataset is not redistributed in this repository. Please follow the access conditions and license specified by the original data provider.

Expected COCO-style structure:

datasets/
└── PigDetect/
    ├── train/
    ├── val/
    ├── test/
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── instances_test.json

5. Installation

The manuscript experiments used:

Component

Version

Operating system

Windows 11

GPU

NVIDIA GeForce RTX 5060

CUDA

12.8

Python

3.10.20

PyTorch

2.11.0

Torchvision

0.26.0

Ultralytics

8.4.34

Create the environment:

conda create -n caq-deim python=3.10.20 -y
conda activate caq-deim
pip install -r requirements.txt

Check PyTorch and CUDA:

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"

6. Configuration

Update the dataset paths in the corresponding YAML file:

num_classes: 1
remap_mscoco_category: false

train_dataloader:
  dataset:
    img_folder: datasets/PigDetect/train
    ann_file: datasets/PigDetect/annotations/instances_train.json

val_dataloader:
  dataset:
    img_folder: datasets/PigDetect/val
    ann_file: datasets/PigDetect/annotations/instances_val.json

Main settings used in the manuscript:

Setting

Value

Epochs

160

Input resolution

960 x 960

Batch size

8

Initial learning rate

0.0004

Weight decay

0.0001

Scheduler

FlatCosine

Initialization

Official pretrained weights

The configuration filenames below are examples. Replace them with the actual filenames in this repository.

7. Training

Train the DEIM baseline:

python train.py -c configs/deim_dfine/deim_hgnetv2_n_pigdetect.yml --use-amp --seed=0

Train CAQ-DEIM:

python train.py -c configs/caq_deim/caq_deim_hgnetv2_n_pigdetect.yml --use-amp --seed=0

Windows PowerShell:

python train.py `
  -c configs/caq_deim/caq_deim_hgnetv2_n_pigdetect.yml `
  --use-amp `
  --seed=0

8. Evaluation

python train.py   -c configs/caq_deim/caq_deim_hgnetv2_n_pigdetect.yml   --test-only   -r path/to/checkpoint.pth

Windows PowerShell:

python train.py `
  -c configs/caq_deim/caq_deim_hgnetv2_n_pigdetect.yml `
  --test-only `
  -r path/to/checkpoint.pth

Reported metrics follow the COCO protocol:

AP: mean AP over IoU thresholds from 0.50 to 0.95;

AP50: AP at IoU = 0.50;

AP75: AP at IoU = 0.75;

AR: average recall.

9. Inference and Model Complexity

Torch inference:

python tools/inference/torch_inf.py   -c configs/caq_deim/caq_deim_hgnetv2_n_pigdetect.yml   -r path/to/checkpoint.pth   --input path/to/image_or_video   --device cuda:0

Calculate parameters and GFLOPs:

python tools/benchmark/get_info.py   -c configs/caq_deim/caq_deim_hgnetv2_n_pigdetect.yml

Final reported model complexity:

Parameters: 4.688 M
GFLOPs:     17.965
Input size: 960 x 960

FPS is not reported in the manuscript because a standardized reproducible inference-speed benchmark was not conducted.

10. Checkpoints and Reproducibility

Model weights should be released through GitHub Releases, Zenodo, Figshare, or an institutional repository rather than committed directly to the Git repository.

Model

Dataset

AP

AP75

Checkpoint

DEIM baseline

PigDetect

0.752

0.853

To be released

CAQ-DEIM

PigDetect

0.775

0.884

To be released

For reproducibility, the repository should include:

CA-DOS complexity-score generation and sampling code;

BCFE implementation;

HQ-MAL implementation;

complete YAML configurations;

training and testing commands;

random seed settings;

evaluation scripts;

external frame manifests and annotation metadata where redistribution is permitted.

11. Citation

The CAQ-DEIM manuscript has not yet received final bibliographic information. Replace the placeholders below after publication:

@article{caqdeim2026,
  title   = {CAQ-DEIM: Complexity-Aware Quality-Guided End-to-End Pig Detection in Complex Pig-Barn Environments},
  author  = {[Author names]},
  journal = {[Journal name]},
  year    = {2026},
  volume  = {[Volume]},
  number  = {[Issue]},
  pages   = {[Article number]},
  doi     = {[DOI]}
}

Please also cite the original DEIM paper:

@inproceedings{huang2025deim,
  title     = {DEIM: DETR with Improved Matching for Fast Convergence},
  author    = {Huang, Shihua and Lu, Zhichao and Cun, Xiaodong and Yu, Yongjun and Zhou, Xiao and Shen, Xi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}

12. License

This repository is derived from the open-source DEIM framework. Please retain all applicable upstream copyright and license notices.

See LICENSE for details.

13. Acknowledgements

This work is built on:

DEIM

D-FINE

RT-DETR

We thank the authors and maintainers of these projects and the providers of the public pig-barn datasets used in this research.

14. Contact

For implementation and reproduction questions, please open an issue:

https://github.com/xingyaunbo/CAQ-DEIM/issues

Author and corresponding-author contact information will be added after the manuscript metadata are finalized.
