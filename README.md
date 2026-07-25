# CAQ-DEIM

## Complexity-Aware Quality-Guided End-to-End Pig Detection

CAQ-DEIM is an end-to-end pig detection model designed for complex group-housed pig-barn environments. It is developed on the basis of DEIM and focuses on improving bounding-box localization in scenes containing dense targets, overlapping pigs, close physical contact, and blurred boundaries.

CAQ-DEIM retains the NMS-free end-to-end inference pipeline of DEIM and introduces three improvements:

* **CA-DOS:** a complexity-aware Dense O2O sampling strategy that increases the training exposure of dense and overlapping pig images.
* **BCFE:** a boundary-context feature enhancement module that strengthens local boundary and contextual feature representation.
* **HQ-MAL:** a high-quality matchability-aware loss that improves consistency between classification confidence and localization quality.

---

## Results

Experiments were conducted on the public PigDetect dataset using an input resolution of `960 × 960`.

| Method       |        AP |      AP75 |  Parameters |     GFLOPs |
| ------------ | --------: | --------: | ----------: | ---------: |
| DEIM         |     0.752 |     0.853 |     3.723 M |     15.092 |
| **CAQ-DEIM** | **0.775** | **0.884** | **4.688 M** | **17.965** |

Compared with the original DEIM baseline, CAQ-DEIM improves:

* AP by **2.3 percentage points**
* AP75 by **3.1 percentage points**

### Ablation Results

| Method               |        AP |      AP75 |
| -------------------- | --------: | --------: |
| DEIM                 |     0.752 |     0.853 |
| DEIM + CA-DOS        |     0.765 |     0.871 |
| DEIM + BCFE          |     0.763 |     0.875 |
| DEIM + CA-DOS + BCFE |     0.769 |     0.872 |
| **CAQ-DEIM**         | **0.775** | **0.884** |

---

## Dataset

The experiments use the public **PigDetect** object-detection dataset from the PigBench benchmark.

| Split      | Images | Bounding Boxes |
| ---------- | -----: | -------------: |
| Training   |  2,431 |         33,197 |
| Validation |    250 |          3,544 |
| Test       |    250 |          5,436 |
| Total      |  2,931 |         42,177 |

Dataset DOI:

```text
https://doi.org/10.25625/I6UYE9
```

The PigDetect dataset is not included in this repository. Please download it from the original data provider and organize it in COCO format.

Example directory structure:

```text
datasets/
└── PigDetect/
    ├── train/
    ├── val/
    ├── test/
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── instances_test.json
```

---

## Installation

Create the environment:

```bash
conda create -n caq-deim python=3.10 -y
conda activate caq-deim
pip install -r requirements.txt
```

The main experimental environment used in the paper was:

* Python 3.10
* PyTorch 2.11
* Torchvision 0.26
* CUDA 12.8
* Windows 11

---

## Training

Modify the dataset paths and training settings in the corresponding YAML configuration file.

Example training command:

```bash
python train.py \
  -c configs/deim_dfine/caq_deim_hgnetv2_n_pigdetect.yml \
  --use-amp \
  --seed=0
```

Windows PowerShell:

```powershell
python train.py `
  -c configs/deim_dfine/caq_deim_hgnetv2_n_pigdetect.yml `
  --use-amp `
  --seed=0
```

Main training settings:

* Input resolution: `960 × 960`
* Training epochs: `160`
* Batch size: `8`
* Initial learning rate: `0.0004`
* Weight decay: `0.0001`

Please replace the configuration path with the actual CAQ-DEIM configuration filename in this repository.

---

## Evaluation

Evaluate a trained checkpoint:

```bash
python train.py \
  -c configs/deim_dfine/caq_deim_hgnetv2_n_pigdetect.yml \
  --test-only \
  -r path/to/checkpoint.pth
```

The reported metrics follow the COCO object-detection evaluation protocol, including AP, AP50, AP75, and AR.

---

## Model Weights

The trained model weights are not stored directly in the Git repository because of GitHub file-size limitations.

Model checkpoints will be released through GitHub Releases or a permanent research repository.

| Model         |    AP |  AP75 | Checkpoint     |
| ------------- | ----: | ----: | -------------- |
| DEIM baseline | 0.752 | 0.853 | To be released |
| CAQ-DEIM      | 0.775 | 0.884 | To be released |

---

## Citation

The CAQ-DEIM paper is currently under preparation. Citation information will be updated after publication.

```bibtex
@article{caqdeim2026,
  title   = {CAQ-DEIM: Complexity-Aware Quality-Guided End-to-End Pig Detection in Complex Pig-Barn Environments},
  author  = {Author names},
  journal = {Journal name},
  year    = {2026}
}
```

Please also cite the original DEIM work:

```bibtex
@inproceedings{huang2025deim,
  title     = {DEIM: DETR with Improved Matching for Fast Convergence},
  author    = {Huang, Shihua and Lu, Zhichao and Cun, Xiaodong and Yu, Yongjun and Zhou, Xiao and Shen, Xi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

---

## Acknowledgements

This repository is developed based on the following open-source projects:

* [DEIM](https://github.com/ShihuaHuang95/DEIM)
* [D-FINE](https://github.com/Peterande/D-FINE)
* [RT-DETR](https://github.com/lyuwenyu/RT-DETR)

We thank the authors and maintainers of these projects.

---

## License

This project follows the applicable license requirements of the original DEIM repository. See the `LICENSE` file for details.

---

## Contact

For questions about the code or experimental reproduction, please open an issue:

```text
https://github.com/xingyaunbo/CAQ-DEIM/issues
```
