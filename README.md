# MLAB_DR_Seg: Retinal Lesion Segmentation on the IDRiD Dataset

A transformer-based framework for automated retinal lesion segmentation from color fundus images using the IDRiD (Indian Diabetic Retinopathy Image Dataset).

## Overview

MLAB_DR_Seg provides an end-to-end deep learning pipeline for retinal lesion segmentation in diabetic retinopathy fundus images.

The framework leverages transformer-based feature extraction and semantic segmentation techniques to identify and segment retinal lesions from retinal fundus photographs. The project supports training, validation, testing, and quantitative evaluation using standard segmentation metrics.

## Features

- End-to-end pipeline for retinal lesion segmentation
- Transformer-based backbone architecture
- Support for IDRiD lesion segmentation dataset
- Configurable training and evaluation pipeline
- Quantitative lesion-wise evaluation
- Prediction visualization and result generation
- Support for pretrained backbone initialization

## Prerequisites

Before proceeding, ensure the following resources are available in your environment.

### Pretrained Backbone Weights

Download the pretrained backbone weights for model initialization from:

https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth

Place the downloaded file at:

```text
<repo_root>/
└── pvt_v2_b3.pth
```

### Dataset Location

Store the IDRiD dataset in the following structure:

```text
<repo_root>/
└── datasets/
    ├── raw/
    │   ├── training/
    │   │   ├── IDRiD_01.jpg
    │   │   ├── IDRiD_01_EX.tif
    │   │   ├── IDRiD_01_HE.tif
    │   │   ├── IDRiD_01_MA.tif
    │   │   ├── IDRiD_01_SE.tif
    │   │   └── ...
    │   │
    │   ├── validation/
    │   │   ├── IDRiD_45.jpg
    │   │   ├── IDRiD_45_EX.tif
    │   │   ├── IDRiD_45_HE.tif
    │   │   ├── IDRiD_45_MA.tif
    │   │   ├── IDRiD_45_SE.tif
    │   │   └── ...
    │   │
    │   └── testing/
    │       ├── IDRiD_55.jpg
    │       ├── IDRiD_55_EX.tif
    │       ├── IDRiD_55_HE.tif
    │       ├── IDRiD_55_MA.tif
    │       ├── IDRiD_55_SE.tif
    │       └── ...
    │
    └── processed/
```

### Dataset Split

- Training Images: 1–44
- Validation Images: 45–54
- Testing Images: 55–81

### Supported Lesion Classes

- Microaneurysms (MA)
- Hemorrhages (HE)
- Hard Exudates (EX)
- Soft Exudates (SE)

## Directory Structure

```text
MLAB_DR_Seg/
├── Data/
│   ├── dataloaders.py
│   └── dataset.py
│
├── Data_prep/
│   └── prepare_dataset.py
│
├── Metrics/
│   ├── losses.py
│   └── performance_metrics.py
│
├── Models/
│   ├── models.py
│   └── pvt_v2.py
│
├── datasets/
│   ├── raw/
│   └── processed/
│
├── outputs/
├── EvalIDRiD/
│
├── train.py
├── eval.py
├── README.md
├── LICENSE
└── pvt_v2_b3.pth
```

## Training and Validation Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Vanshika-P/MLARB_DR_Seg.git
cd MLAB_DR_Seg
```

### 2. Create Environment

```bash
conda create -n mlabdr python=3.10
conda activate mlabdr
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset

Generate the processed dataset by running:

```bash
python Data_prep/prepare_dataset.py
```

### 5. Train

```bash
python train.py
```

Model checkpoints and logs will be saved in:

```text
./outputs/
```

## Evaluation

Run evaluation using:

```bash
python eval.py
```

Evaluation metrics include:

- IoU (Intersection over Union)
- Dice Score
- Precision
- Recall
- Specificity

Prediction outputs and evaluation results are saved in:

```text
./EvalIDRiD/
```

## Results

The framework is evaluated on retinal lesion segmentation tasks using the IDRiD dataset.

Current baseline results:

| Lesion | IoU | Dice | Precision | Recall | Specificity |
|---------|---------|---------|---------|---------|---------|
| MA | 0.4452 | 0.6124 | 0.6382 | 0.5972 | 0.9995 |
| HE | 0.5391 | 0.6938 | 0.6826 | 0.7123 | 0.9976 |

## Author

**Vanshika Patil**

M.Tech (Software Engineering)

Maulana Azad National Institute of Technology (MANIT), Bhopal

## License

This project is intended for academic and research purposes.
