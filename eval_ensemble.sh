#!/bin/bash

# Ensemble Expert Evaluation Script for ChestX-ray14 Dataset

# Dataset and checkpoint paths (MODIFY THESE TO YOUR PATHS)
DATASET_DIR='path/to/ChestXray14/images'
VAL_LIST='datasets/data_splits/cxr14/val_official.txt'
TEST_LIST='datasets/data_splits/cxr14/test_official.txt'

# Model checkpoints (MODIFY THESE TO YOUR CHECKPOINT PATHS)
EVA_X_CKPT='path/to/eva_x_tiny_patch16_cxr14_ft.pth'
MGCA_CKPT='path/to/resnet50_mgca_pt_cxr14_ft.pth'
MEDICAL_MAE_CKPT='path/to/densenet121_medical_mae_pt_cxr14_ft.pth'

# Output directory
SAVE_DIR='./output/cxr14/ensemble_experts'

echo "========================================"
echo "Ensemble Expert Evaluation on ChestX-ray14"
echo "========================================"
echo ""
echo "Models:"
echo "  1. EVA-X-Tiny (EVA02-Tiny/16)"
echo "  2. MGCA (ResNet50)"
echo "  3. Medical MAE (DenseNet121)"
echo ""
echo "Ensemble Strategy:"
echo "  - Validation-based weight tuning"
echo "  - Model-level weights (expert specialization)"
echo "  - Class-level confidence weighting"
echo "  - Sample-level dynamic weighting"
echo "  - Unsupervised adaptive weighting"
echo ""
echo "Validation set: ${VAL_LIST}"
echo "Test set: ${TEST_LIST}"
echo "Output: ${SAVE_DIR}"
echo "========================================"
echo ""

# Create output directory
mkdir -p ${SAVE_DIR}

# Run ensemble evaluation with validation-based weight tuning
echo "Step 1: Tuning weights on validation set..."
echo "Step 2: Evaluating on test set with tuned weights..."
echo ""

python eval_ensemble.py \
    --eva_x_checkpoint ${EVA_X_CKPT} \
    --mgca_checkpoint ${MGCA_CKPT} \
    --medical_mae_checkpoint ${MEDICAL_MAE_CKPT} \
    --data_path ${DATASET_DIR} \
    --val_list ${VAL_LIST} \
    --test_list ${TEST_LIST} \
    --output_dir ${SAVE_DIR} \
    --batch_size 128 \
    --input_size 224 \
    --nb_classes 14 \
    --num_workers 4 \
    --device cuda \
    --seed 42 \
    --dataset chestxray \
    --model eva02_tiny_patch16 \
    --build_timm_transform \
    --aa 'rand-m6-mstd0.5-inc1' \
    --reprob 0.0 \
    --remode pixel \
    --recount 1 \
    --tune_on_val \
    --save_tuned_weights \
    --evaluate_individual

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results saved to: ${SAVE_DIR}/results.json"
echo "Predictions saved to: ${SAVE_DIR}/ensemble_y_pred.npy"
echo ""
