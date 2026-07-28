# OPERA
This repository is the official implementation of the paper “OPERA: Offline Policy-guided Expert Routing and Adaptation for Universal Biomedical Image Analysis”. This repository implements a multi-level ensemble approach for biomedical image classification that combines three expert models without requiring additional training.

## Overview

The ensemble combines three pre-trained expert models:

1. **EVA-X-Tiny** (EVA02-Tiny/16) - Vision Transformer architecture
2. **MGCA** (ResNet50) - Convolutional Neural Network
3. **Medical MAE** (DenseNet121) - Dense Convolutional Network

### Key Features

- **No Training Required**: Uses frozen pre-trained models with learnable ensemble weights
- **Multi-Level Weighting**:
  - Model-level: Expert specialization weights
  - Class-level: Per-class confidence weighting
  - Sample-level: Instance-specific dynamic weighting
  - Adaptive weighting: Unsupervised agreement and uncertainty-based weighting
- **Validation-Based Tuning**: Performance-based weight optimization
- **Comprehensive Metrics**: ROC-AUC, Average Precision, Accuracy per class

## Installation

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)

### Setup

```bash
# Create conda environment (recommended)
conda create -n opera python=3.8
conda activate opera

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Download the ChestX-ray14 dataset:
```bash
# Download from NIH Clinical Center
# https://nihcc.app.box.com/v/ChestXray-NIHCC

# Extract images to a directory
# Update DATASET_DIR in eval_ensemble.sh to point to your images directory
```

The dataset split files are provided in `datasets/data_splits/cxr14/`:
- `train_official.txt` - Training set file list
- `val_official.txt` - Validation set file list
- `test_official.txt` - Test set file list

Each line in the split files follows the format:
```
image_name.png label1 label2 ... label14
```

### Model Checkpoints

You need to provide three pre-trained model checkpoints:

1. **EVA-X-Tiny ViT**: Fine-tuned EVA02-Tiny/16 on ChestX-ray14
2. **MGCA ResNet50**: MGCA pre-trained ResNet50 fine-tuned on ChestX-ray14
3. **Medical MAE DenseNet121**: Medical MAE pre-trained DenseNet121 fine-tuned on ChestX-ray14

Update the checkpoint paths in `eval_ensemble.sh`:
```bash
EVA_X_CKPT='path/to/eva_x_tiny_patch16_cxr14_ft.pth'
MGCA_CKPT='path/to/resnet50_mgca_pt_cxr14_ft.pth'
MEDICAL_MAE_CKPT='path/to/densenet121_medical_mae_pt_cxr14_ft.pth'
```

## Usage

### Quick Start

Run the ensemble evaluation on ChestX-ray14:

```bash
# Edit eval_ensemble.sh to set your paths
vim eval_ensemble.sh

# Run evaluation
bash eval_ensemble.sh
```

### Detailed Usage

```bash
python eval_ensemble.py \
    --eva_x_checkpoint <path-to-eva-x-checkpoint> \
    --mgca_checkpoint <path-to-mgca-checkpoint> \
    --medical_mae_checkpoint <path-to-medical-mae-checkpoint> \
    --data_path <path-to-dataset-images> \
    --val_list datasets/data_splits/cxr14/val_official.txt \
    --test_list datasets/data_splits/cxr14/test_official.txt \
    --output_dir ./output/cxr14/ensemble \
    --batch_size 128 \
    --input_size 224 \
    --nb_classes 14 \
    --num_workers 4 \
    --device cuda \
    --seed 42 \
    --dataset chestxray \
    --model eva02_tiny_patch16 \
    --build_timm_transform \
    --tune_on_val \
    --save_tuned_weights \
    --evaluate_individual
```

### Key Arguments

- `--eva_x_checkpoint`: Path to EVA-X-Tiny checkpoint (required)
- `--mgca_checkpoint`: Path to MGCA ResNet50 checkpoint (required)
- `--medical_mae_checkpoint`: Path to Medical MAE DenseNet121 checkpoint (required)
- `--data_path`: Directory containing dataset images (required)
- `--val_list`: Path to validation split file (required if using `--tune_on_val`)
- `--test_list`: Path to test split file (required)
- `--output_dir`: Directory to save results (default: `./output`)
- `--batch_size`: Batch size for evaluation (default: 64)
- `--input_size`: Input image size (default: 224)
- `--nb_classes`: Number of classes (default: 14 for ChestX-ray14)
- `--tune_on_val`: Enable validation-based weight tuning
- `--save_tuned_weights`: Save tuned weights for later use
- `--evaluate_individual`: Also evaluate individual models and oracle performance

## Output

The evaluation produces:

1. **results.json**: Detailed metrics including:
   - Per-class ROC-AUC, Average Precision, Accuracy
   - Mean metrics across all classes
   - Individual model performances (if `--evaluate_individual` is used)
   - Oracle performance (best single model per sample)

2. **ensemble_y_pred.npy**: Ensemble predictions (numpy array)

3. **tuned_weights.pth**: Saved tuned ensemble weights (if `--save_tuned_weights` is used)

4. **log_<timestamp>.txt**: Detailed execution log

## Repository Structure

```
.
├── models/                      # Model architectures
│   ├── models_eva.py           # EVA02 Vision Transformer
│   ├── models_vit.py           # Standard Vision Transformer
│   └── rope.py                 # Rotary Position Embeddings
├── utils/                       # Utility functions
│   ├── datasets.py             # Dataset builders
│   ├── dataloader_med.py       # Medical imaging dataloaders
│   ├── eva_utils.py            # EVA model utilities
│   ├── misc.py                 # Miscellaneous utilities
│   ├── augment.py              # Data augmentation
│   ├── custom_transforms.py    # Custom transforms
│   ├── pos_embed.py            # Position embeddings
│   ├── lr_sched.py             # Learning rate schedulers
│   ├── lr_decay.py             # Layer-wise LR decay
│   ├── sampler.py              # Data samplers
│   ├── multi_label_loss.py     # Multi-label losses
│   └── lars.py                 # LARS optimizer
├── engines/                     # Training/evaluation engines
│   └── engine_finetune.py      # Fine-tuning engine
├── datasets/                    # Dataset configurations
│   └── data_splits/            # Train/val/test splits
│       └── cxr14/              # ChestX-ray14 splits
├── train_ensemble.py            # Main ensemble script
├── eval_ensemble.sh             # Evaluation script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Ensemble Architecture

### EnsembleExpertModel

The core ensemble class implements a sophisticated multi-level weighting system:

1. **Model-Level Weights**: Learnable parameters that balance the three expert models based on overall performance

2. **Class-Level Attention**: Per-class attention weights allowing each model to specialize in different disease categories

3. **Sample-Level Dynamic Weighting**: MLP network that adapts weights based on individual sample characteristics

4. **Confidence-Based Weighting**: Weights predictions by model confidence scores

5. **Adaptive Weighting**: Unsupervised weighting based on:
   - Prediction agreement between models
   - Prediction uncertainty metrics
   - Entropy and variance analysis

### Weight Tuning Strategies

Three strategies are available:

1. **Performance-Based (Recommended)**: Analytical solution based on per-model, per-class AUC on validation set
2. **Supervised**: Gradient-based optimization with validation labels
3. **Unsupervised**: Iterative optimization using agreement and uncertainty metrics

## License

This project is released under the MIT License.

## Acknowledgments

This code builds upon several excellent open-source projects:

- [EVA-02](https://github.com/baaivision/EVA) - Vision Transformer backbone
- [timm](https://github.com/rwightman/pytorch-image-models) - PyTorch Image Models
- [DeiT](https://github.com/facebookresearch/deit) - Data-efficient Image Transformers
- [MAE](https://github.com/facebookresearch/mae) - Masked Autoencoders
