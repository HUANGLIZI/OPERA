"""
This script implements a multi-level ensemble approach combining:
- EVA-X-Tiny (ViT-Tiny/16)
- MGCA (ResNet50)
- Medical MAE (DenseNet121)

The ensemble uses three levels of weight adjustment:
1. Model-level: Different architecture preferences (expert specializations)
2. Class-level: Per-class confidence weighting based on prediction probabilities
3. Sample-level: Instance-specific dynamic weighting

The goal is to achieve near-optimal performance where the ensemble is correct
whenever at least one model is correct.
"""

import os
import re
import time
import json
import torch
import argparse
import datetime
import numpy as np
import sys
from pathlib import Path
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from timm.models import create_model
from torchvision import models as torchvision_models
import math

import utils.misc as misc
from utils import build_dataset_chest_xray
from models import models_vit, models_eva
from engines import evaluate_chestxray


class Logger(object):
    """Logger class to save output to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class EnsembleExpertModel(torch.nn.Module):
    """
    Multi-level ensemble model combining three expert models with:
    1. Model-level learnable weights
    2. Class-level confidence-based weighting
    3. Sample-level dynamic weighting
    4. Feature-level fusion
    """

    def __init__(self, model_configs, num_classes=14, feature_dim=512, enable_feature_fusion=False,
                 tuned_weights_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_models = len(model_configs)
        self.feature_dim = feature_dim
        self.enable_feature_fusion = enable_feature_fusion
        self.tuned_weights_path = tuned_weights_path

        self.register_buffer('original_tuned_model_weights', None)
        self.register_buffer('original_tuned_class_attention', None)

        self.models = torch.nn.ModuleList()
        self.model_names = []
        self.model_types = []

        for config in model_configs:
            model = self._load_model(config)
            self.models.append(model)
            self.model_names.append(config['name'])
            self.model_types.append(config['type'])

        self.model_weights = torch.nn.Parameter(
            torch.ones(self.num_models) / self.num_models
        )

        self.class_attention = torch.nn.Parameter(
            torch.ones(self.num_models, num_classes) / self.num_models
        )

        self.sample_weight_net = torch.nn.Sequential(
            torch.nn.Linear(self.num_models * num_classes, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, self.num_models),
            torch.nn.Softmax(dim=-1)
        )

        self.confidence_temperature = torch.nn.Parameter(torch.tensor(2.0))

        self.enable_uncertainty_weighting = True
        self.enable_agreement_weighting = True
        self.enable_temperature_scaling = True

        self.register_buffer('temperature_per_model', torch.ones(self.num_models))
        self.register_buffer('temperature_per_class', torch.ones(self.num_models, num_classes))

        if self.enable_feature_fusion:
            actual_feature_dim = 192 + 2048 + 1024
            self.feature_fusion = torch.nn.Sequential(
                torch.nn.Linear(actual_feature_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, num_classes)
            )
            self.fusion_gate = torch.nn.Parameter(torch.tensor(0.5))
        else:
            self.feature_fusion = None
            self.fusion_gate = None

        if tuned_weights_path is not None and os.path.exists(tuned_weights_path):
            self.load_tuned_weights(tuned_weights_path)

    def load_tuned_weights(self, path):
        """Load pre-tuned weights from validation set"""
        print(f"\nLoading tuned weights from: {path}")
        checkpoint = torch.load(path, map_location='cpu')

        if 'model_weights' in checkpoint:
            self.model_weights.data = checkpoint['model_weights']
            print(f"✓ Loaded model_weights: {F.softmax(self.model_weights, dim=0).numpy()}")

        if 'class_attention' in checkpoint:
            self.class_attention.data = checkpoint['class_attention']
            print(f"✓ Loaded class_attention")

        if 'confidence_temperature' in checkpoint:
            self.confidence_temperature.data = checkpoint['confidence_temperature']
            print(f"✓ Loaded confidence_temperature: {self.confidence_temperature.item():.4f}")

        if 'temperature_per_model' in checkpoint:
            self.temperature_per_model.data = checkpoint['temperature_per_model']
            print(f"✓ Loaded temperature_per_model: {self.temperature_per_model.numpy()}")

        if 'temperature_per_class' in checkpoint:
            self.temperature_per_class.data = checkpoint['temperature_per_class']
            print(f"✓ Loaded temperature_per_class")

        print("Successfully loaded all tuned weights!\n")

    def save_tuned_weights(self, path):
        """Save tuned weights for later use"""
        checkpoint = {
            'model_weights': self.model_weights.data.cpu(),
            'class_attention': self.class_attention.data.cpu(),
            'confidence_temperature': self.confidence_temperature.data.cpu(),
            'temperature_per_model': self.temperature_per_model.cpu(),
            'temperature_per_class': self.temperature_per_class.cpu(),
        }
        torch.save(checkpoint, path)
        print(f"✓ Saved tuned weights to: {path}")

    def _load_model(self, config):
        """Load a pre-trained model based on configuration"""
        model_type = config['type']
        checkpoint_path = config['checkpoint']
        num_classes = config['num_classes']

        if model_type == 'vit':
            model_name = config['model_name']

            if 'eva' in model_name.lower():
                model_fn = getattr(models_eva, model_name, None)
                if model_fn is None:
                    raise ValueError(f"EVA model {model_name} not found in models_eva")
                model = model_fn(
                    pretrained=False,
                    img_size=config.get('input_size', 224),
                    num_classes=num_classes,
                    drop_rate=0.0,
                    drop_path_rate=0.0,
                    attn_drop_rate=0.0,
                    use_mean_pooling=True,
                )
            else:
                model_fn = getattr(models_vit, model_name, None)
                if model_fn is not None:
                    model = model_fn(
                        img_size=config.get('input_size', 224),
                        num_classes=num_classes,
                        global_pool=config.get('global_pool', False),
                    )
                else:
                    model = create_model(
                        model_name,
                        pretrained=False,
                        img_size=config.get('input_size', 224),
                        num_classes=num_classes,
                        drop_rate=0.0,
                        drop_path_rate=0.0,
                        attn_drop_rate=0.0,
                    )
        elif model_type == 'resnet50':
            model = torchvision_models.resnet50(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)
        elif model_type == 'densenet121':
            model = torchvision_models.densenet121(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if model_type == 'resnet50' and any('img_encoder_q.model.' in k for k in state_dict.keys()):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'img_encoder_q.model.' in k:
                    new_state_dict[k.replace('img_encoder_q.model.', '')] = v
            state_dict = new_state_dict

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {config['name']}: {msg}")

        for param in model.parameters():
            param.requires_grad = False

        return model

    def extract_features(self, x, model, model_type):
        """Extract intermediate features from models"""
        try:
            if model_type == 'vit':
                if hasattr(model, 'forward_features'):
                    features = model.forward_features(x)
                    if len(features.shape) == 3:
                        features = features.mean(1)
                    elif len(features.shape) == 2:
                        pass
                    if hasattr(model, 'fc_norm') and model.global_pool:
                        features = model.fc_norm(features)
                    return features
                else:
                    return None
            elif model_type in ['resnet50', 'densenet121']:
                if model_type == 'resnet50':
                    x = model.conv1(x)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)
                    x = model.avgpool(x)
                    features = torch.flatten(x, 1)
                else:
                    features = model.features(x)
                    features = F.relu(features, inplace=True)
                    features = F.adaptive_avg_pool2d(features, (1, 1))
                    features = torch.flatten(features, 1)
                return features
        except Exception as e:
            print(f"Warning: Feature extraction failed for {model_type}: {e}")
            return None
        return None

    def compute_prediction_agreement(self, probs_stack):
        """
        Compute agreement score between model predictions (unsupervised)

        Args:
            probs_stack: [num_models, batch_size, num_classes]

        Returns:
            agreement_scores: [batch_size] - higher means models agree more
        """
        probs = probs_stack.permute(1, 0, 2)
        batch_size = probs.shape[0]

        agreement_scores = []
        for b in range(batch_size):
            sample_preds = probs[b]

            similarities = []
            for i in range(self.num_models):
                for j in range(i + 1, self.num_models):
                    pred_i = sample_preds[i].unsqueeze(0)
                    pred_j = sample_preds[j].unsqueeze(0)

                    norm_i = torch.norm(pred_i)
                    norm_j = torch.norm(pred_j)

                    if norm_i < 1e-8 or norm_j < 1e-8:
                        sim = torch.tensor(1.0, device=pred_i.device)
                    else:
                        sim = F.cosine_similarity(pred_i, pred_j)

                    similarities.append(sim)

            avg_similarity = torch.stack(similarities).mean()
            agreement_scores.append(avg_similarity)

        return torch.stack(agreement_scores)

    def compute_prediction_uncertainty(self, probs_stack):
        """
        Compute uncertainty metrics for predictions (unsupervised)

        Args:
            probs_stack: [num_models, batch_size, num_classes]

        Returns:
            entropy: [batch_size, num_models] - prediction entropy per model
            variance: [batch_size, num_classes] - variance across models
        """
        eps = 1e-8

        entropy = -(probs_stack * torch.log(probs_stack + eps) +
                    (1 - probs_stack) * torch.log(1 - probs_stack + eps))
        entropy = entropy.mean(dim=2)
        entropy = entropy.permute(1, 0)

        variance = probs_stack.var(dim=0)

        return entropy, variance

    def compute_confidence_gap(self, probs):
        """
        Compute confidence gap (difference between top-2 predictions)

        Args:
            probs: [batch_size, num_classes]

        Returns:
            gap: [batch_size] - higher means more confident
        """
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)

        if sorted_probs.shape[1] > 1:
            gap = sorted_probs[:, 0] - sorted_probs[:, 1]
        else:
            gap = sorted_probs[:, 0]

        return gap

    def adaptive_weight_computation(self, probs_stack, use_base_weights=True):
        """
        Compute adaptive weights with NaN detection and safe fallback

        Strategy:
        - Use model_weights as base preference
        - High agreement -> Use base weights
        - High disagreement -> Trust most confident model
        - Medium cases -> Weight by extremity with base bias
        - NaN detection: If any computation fails, fallback to base weights

        Args:
            probs_stack: [num_models, batch_size, num_classes]
            use_base_weights: If True, use model_weights as base (default: True)

        Returns:
            adaptive_weights: [batch_size, num_models] - confidence scores for each model per sample
        """
        batch_size = probs_stack.shape[1]
        device = probs_stack.device

        if use_base_weights:
            base_weights = F.softmax(self.model_weights, dim=0)
        else:
            base_weights = torch.ones(self.num_models, device=device) / self.num_models

        adaptive_weights = base_weights.unsqueeze(0).repeat(batch_size, 1)

        try:
            agreement = self.compute_prediction_agreement(probs_stack)

            if torch.isnan(agreement).any():
                print(f"Warning: NaN detected in agreement scores for {torch.isnan(agreement).sum()} samples")
                agreement = torch.where(torch.isnan(agreement),
                                       torch.tensor(0.5, device=device),
                                       agreement)

            entropy, variance = self.compute_prediction_uncertainty(probs_stack)

            if torch.isnan(entropy).any():
                print(f"Warning: NaN detected in entropy for {torch.isnan(entropy).any(dim=1).sum()} samples")
                entropy = torch.where(torch.isnan(entropy),
                                     torch.tensor(0.5, device=device),
                                     entropy)

            if torch.isnan(variance).any():
                print(f"Warning: NaN detected in variance for {torch.isnan(variance).any(dim=1).sum()} samples")
                variance = torch.where(torch.isnan(variance),
                                      torch.tensor(0.05, device=device),
                                      variance)

            nan_sample_count = 0

            for b in range(batch_size):
                try:
                    sample_agreement = agreement[b].item()
                    sample_variance = variance[b].mean().item()
                    sample_entropy = entropy[b]

                    import math
                    if math.isnan(sample_agreement) or math.isnan(sample_variance) or torch.isnan(sample_entropy).any():
                        nan_sample_count += 1
                        continue

                    if sample_agreement > 0.90 and sample_variance < 0.03:
                        adaptive_weights[b] = base_weights
                    elif sample_agreement < 0.60 or sample_variance > 0.12:
                        inv_entropy = 1.0 / (sample_entropy + 1e-6)

                        if not torch.isnan(inv_entropy).any():
                            weights = F.softmax(inv_entropy * 2.0, dim=0)
                            if not torch.isnan(weights).any():
                                adaptive_weights[b] = weights
                            else:
                                nan_sample_count += 1
                    else:
                        extremity = torch.zeros(self.num_models, device=device)
                        for m in range(self.num_models):
                            model_probs = probs_stack[m, b, :]
                            extremity[m] = (model_probs - 0.5).abs().mean()

                        if not torch.isnan(extremity).any():
                            extremity_weights = F.softmax(extremity * 3.0, dim=0)

                            if not torch.isnan(extremity_weights).any():
                                adaptive_weights[b] = 0.7 * extremity_weights + 0.3 * base_weights
                            else:
                                nan_sample_count += 1
                        else:
                            nan_sample_count += 1

                except Exception as e:
                    nan_sample_count += 1
                    continue

            if nan_sample_count > 0:
                print(f"Info: {nan_sample_count}/{batch_size} samples used base_weights due to NaN/errors")

        except Exception as e:
            print(f"Warning: Adaptive weight computation failed: {e}")
            print("Falling back to base weights for all samples")

        return adaptive_weights

    def apply_unsupervised_temperature_scaling(self, probs_stack):
        """
        Apply unsupervised temperature scaling based on prediction sharpness

        Args:
            probs_stack: [num_models, batch_size, num_classes]

        Returns:
            calibrated_probs: [num_models, batch_size, num_classes]
        """
        if not self.enable_temperature_scaling:
            return probs_stack

        calibrated_probs = []

        for m in range(self.num_models):
            model_probs = probs_stack[m]

            sharpness = (model_probs - 0.5).abs().mean().item()

            if sharpness > 0.4:
                T = 1.5
            elif sharpness < 0.1:
                T = 0.7
            else:
                T = 1.0

            self.temperature_per_model[m] = T

            eps = 1e-7
            model_probs_clamped = torch.clamp(model_probs, eps, 1 - eps)
            logits = torch.log(model_probs_clamped / (1 - model_probs_clamped))
            calibrated = torch.sigmoid(logits / T)

            calibrated_probs.append(calibrated)

        return torch.stack(calibrated_probs, dim=0)

    def forward(self, x, return_details=False):
        """
        Forward pass with multi-level ensemble

        Args:
            x: Input images [B, C, H, W]
            return_details: If True, return detailed information about weights

        Returns:
            logits: Final ensemble logits [B, num_classes]
            details: (optional) Dictionary with weight information
        """
        batch_size = x.size(0)

        all_logits = []
        all_probs = []
        all_features = []

        for i, (model, model_type) in enumerate(zip(self.models, self.model_types)):
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)

                all_logits.append(logits)
                all_probs.append(probs)

                if self.enable_feature_fusion:
                    features = self.extract_features(x, model, model_type)
                    if features is not None:
                        all_features.append(features)

        logits_stack = torch.stack(all_logits, dim=0)
        probs_stack = torch.stack(all_probs, dim=0)

        if self.enable_temperature_scaling:
            probs_stack = self.apply_unsupervised_temperature_scaling(probs_stack)

        model_weights_norm = F.softmax(self.model_weights, dim=0)

        class_attention_norm = F.softmax(self.class_attention, dim=0)

        probs_concat = probs_stack.permute(1, 0, 2).reshape(batch_size, -1)
        sample_weights = self.sample_weight_net(probs_concat)

        confidence_scores = torch.max(probs_stack, dim=2)[0]
        temperature = torch.clamp(self.confidence_temperature, 0.1, 10.0)
        confidence_weights = F.softmax(confidence_scores.permute(1, 0) * temperature, dim=1)

        adaptive_weights = None
        if self.enable_uncertainty_weighting or self.enable_agreement_weighting:
            adaptive_weights = self.adaptive_weight_computation(probs_stack)

        model_w = model_weights_norm.view(1, 1, self.num_models)
        class_w = class_attention_norm.t().unsqueeze(0)
        sample_w = sample_weights.unsqueeze(1)
        confidence_w = confidence_weights.unsqueeze(1)

        if adaptive_weights is not None:
            adaptive_w = adaptive_weights.unsqueeze(1)
            final_weights = model_w * class_w * sample_w * confidence_w * adaptive_w
        else:
            final_weights = model_w * class_w * sample_w * confidence_w

        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)

        ensemble_probs = (probs_stack.permute(1, 2, 0) * final_weights).sum(dim=-1)

        if self.enable_feature_fusion and len(all_features) == self.num_models:
            features_concat = torch.cat(all_features, dim=1)
            feature_logits = self.feature_fusion(features_concat)
            feature_probs = torch.sigmoid(feature_logits)

            gate_weight = torch.sigmoid(self.fusion_gate)
            ensemble_probs = gate_weight * ensemble_probs + (1 - gate_weight) * feature_probs

        ensemble_probs = torch.clamp(ensemble_probs, 1e-7, 1 - 1e-7)
        ensemble_logits = torch.log(ensemble_probs / (1 - ensemble_probs))

        if return_details:
            details = {
                'model_weights': model_weights_norm,
                'class_attention': class_attention_norm,
                'sample_weights': sample_weights,
                'confidence_weights': confidence_weights,
                'adaptive_weights': adaptive_weights,
                'temperature_per_model': self.temperature_per_model,
                'final_weights': final_weights,
                'individual_probs': probs_stack,
            }
            return ensemble_logits, details

        return ensemble_logits


def tune_weights_on_validation_performance_based(data_loader, model, device, args):
    """
    Tune ensemble weights based on validation performance using labels and predictions.

    This method computes optimal weights by analyzing which model performs best on validation:
    1. Collect predictions and labels on validation set
    2. Compute per-model, per-class AUC/accuracy
    3. Assign higher weights to better-performing models
    4. No gradient descent - direct analytical solution based on performance

    Args:
        data_loader: Validation data loader (with labels)
        model: Ensemble model
        device: Device to run on
        args: Arguments

    Returns:
        Best tuned weights based on validation performance
    """
    print("\n" + "="*60)
    print("Tuning Weights Based on Validation Performance")
    print("="*60)
    print(f"Validation samples: {len(data_loader.dataset)}")
    print("Strategy: Assign weights based on per-model, per-class AUC")
    print()

    model.eval()

    print("Step 1: Collecting predictions and labels from validation set...")
    all_individual_probs = []
    all_targets = []

    for batch_idx, batch in enumerate(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            batch_probs = []
            for individual_model in model.models:
                logits = individual_model(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs)

            batch_probs_stack = torch.stack(batch_probs, dim=0)
            all_individual_probs.append(batch_probs_stack)
            all_targets.append(target)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")

    all_probs = torch.cat(all_individual_probs, dim=1).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    print(f"✓ Collected predictions: {all_probs.shape}")
    print(f"✓ Collected targets: {all_targets.shape}")

    print("\nStep 2: Computing per-model, per-class AUC and Accuracy on validation set...")

    class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                   'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                   'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                   'Pleural_Thickening', 'Hernia']

    performance_matrix = np.zeros((model.num_models, model.num_classes))
    accuracy_matrix = np.zeros((model.num_models, model.num_classes))
    map_matrix = np.zeros((model.num_models, model.num_classes))

    print("\nPer-model, per-class validation AUC:")
    print("-" * 80)
    print(f"{'Class':<20} | {'EVA-Tiny':>8} | {'MGCA':>8} | {'Med-MAE':>8} | {'Best Model':<20}")
    print("-" * 80)

    for c in range(model.num_classes):
        class_targets = all_targets[:, c]
        class_aucs = []

        for m in range(model.num_models):
            model_preds = all_probs[m, :, c]

            try:
                auc = roc_auc_score(class_targets, model_preds)
                performance_matrix[m, c] = auc
                class_aucs.append(auc)
            except:
                performance_matrix[m, c] = 0.5
                class_aucs.append(0.5)

            binary_preds = (model_preds > 0.5).astype(int)
            acc = accuracy_score(class_targets, binary_preds)
            accuracy_matrix[m, c] = acc

            try:
                ap = average_precision_score(class_targets, model_preds)
                map_matrix[m, c] = ap
            except:
                map_matrix[m, c] = 0.0

        best_model_idx = np.argmax(class_aucs)
        best_model_name = model.model_names[best_model_idx]

        print(f"{class_names[c]:<20} | {class_aucs[0]:>8.4f} | {class_aucs[1]:>8.4f} | "
              f"{class_aucs[2]:>8.4f} | {best_model_name:<20}")

    print("-" * 80)

    print("\nPer-model, per-class validation Accuracy:")
    print("-" * 80)
    print(f"{'Class':<20} | {'EVA-Tiny':>8} | {'MGCA':>8} | {'Med-MAE':>8} | {'Best Model':<20}")
    print("-" * 80)

    for c in range(model.num_classes):
        class_accs = [accuracy_matrix[m, c] for m in range(model.num_models)]
        best_model_idx = np.argmax(class_accs)
        best_model_name = model.model_names[best_model_idx]

        print(f"{class_names[c]:<20} | {class_accs[0]:>8.4f} | {class_accs[1]:>8.4f} | "
              f"{class_accs[2]:>8.4f} | {best_model_name:<20}")

    print("-" * 80)

    print("\nPer-model, per-class validation mAP (Average Precision):")
    print("-" * 80)
    print(f"{'Class':<20} | {'EVA-Tiny':>8} | {'MGCA':>8} | {'Med-MAE':>8} | {'Best Model':<20}")
    print("-" * 80)

    for c in range(model.num_classes):
        class_maps = [map_matrix[m, c] for m in range(model.num_models)]
        best_model_idx = np.argmax(class_maps)
        best_model_name = model.model_names[best_model_idx]

        print(f"{class_names[c]:<20} | {class_maps[0]:>8.4f} | {class_maps[1]:>8.4f} | "
              f"{class_maps[2]:>8.4f} | {best_model_name:<20}")

    print("-" * 80)
    sys.stdout.flush()

    print("\nStep 3: Computing model-level weights based on average AUC...")

    model_avg_aucs = performance_matrix.mean(axis=1)
    model_avg_accs = accuracy_matrix.mean(axis=1)
    model_avg_maps = map_matrix.mean(axis=1)

    print("\nAverage validation AUC per model:")
    for m, (name, auc) in enumerate(zip(model.model_names, model_avg_aucs)):
        print(f"  {name:25s}: {auc:.4f}")

    print("\nAverage validation Accuracy per model:")
    for m, (name, acc) in enumerate(zip(model.model_names, model_avg_accs)):
        print(f"  {name:25s}: {acc:.4f}")

    print("\nAverage validation mAP per model:")
    for m, (name, map_val) in enumerate(zip(model.model_names, model_avg_maps)):
        print(f"  {name:25s}: {map_val:.4f}")
    sys.stdout.flush()

    temperature = 10.0
    model_scores = model_avg_aucs * temperature
    model_weights_norm = np.exp(model_scores) / np.exp(model_scores).sum()

    print("\nComputed model-level weights (based on performance):")
    for m, (name, weight) in enumerate(zip(model.model_names, model_weights_norm)):
        print(f"  {name:25s}: {weight:.4f}")

    print("\nStep 4: Computing class-level weights based on per-class performance...")

    class_weights_matrix = np.zeros_like(performance_matrix)
    for c in range(model.num_classes):
        class_perfs = performance_matrix[:, c]
        class_scores = class_perfs * temperature
        class_weights = np.exp(class_scores) / np.exp(class_scores).sum()
        class_weights_matrix[:, c] = class_weights

    print("\nClass-level specialization (top model for each class):")
    for c in range(model.num_classes):
        top_model_idx = np.argmax(class_weights_matrix[:, c])
        top_weight = class_weights_matrix[top_model_idx, c]
        print(f"  {class_names[c]:<20}: {model.model_names[top_model_idx]:<25} "
              f"(weight={top_weight:.3f}, AUC={performance_matrix[top_model_idx, c]:.4f})")

    print("\nStep 5: Computing optimal temperature parameters...")

    temperature_per_model = np.ones(model.num_models)
    for m in range(model.num_models):
        avg_auc = model_avg_aucs[m]
        if avg_auc > 0.85:
            T = 1.0
        elif avg_auc > 0.80:
            T = 1.1
        elif avg_auc > 0.75:
            T = 1.2
        else:
            T = 0.9
        temperature_per_model[m] = T

    print("\nTemperature per model:")
    for m, (name, T) in enumerate(zip(model.model_names, temperature_per_model)):
        status = "Smooth" if T > 1.05 else ("Sharpen" if T < 0.95 else "Normal")
        print(f"  {name:25s}: T={T:.2f} ({status})")

    overall_agreement = 0.0
    count = 0
    for c in range(model.num_classes):
        aucs = performance_matrix[:, c]
        if aucs.std() < 0.05:
            overall_agreement += 1.0
        count += 1

    agreement_ratio = overall_agreement / count
    if agreement_ratio > 0.75:
        conf_temp = 1.5
    elif agreement_ratio > 0.5:
        conf_temp = 2.0
    else:
        conf_temp = 2.5

    print(f"\nGlobal confidence temperature: {conf_temp:.2f}")
    print(f"  (based on model agreement: {agreement_ratio:.2f})")

    print("\nStep 6: Applying optimized weights to model...")

    best_weights = {
        'model_weights': torch.log(torch.tensor(model_weights_norm, dtype=torch.float32) + 1e-8),
        'class_attention': torch.log(torch.tensor(class_weights_matrix, dtype=torch.float32) + 1e-8),
        'confidence_temperature': torch.tensor(conf_temp, dtype=torch.float32),
        'temperature_per_model': torch.tensor(temperature_per_model, dtype=torch.float32),
    }

    with torch.no_grad():
        model.model_weights.data = best_weights['model_weights'].to(device)
        model.class_attention.data = best_weights['class_attention'].to(device)
        model.confidence_temperature.data = best_weights['confidence_temperature'].to(device)
        model.temperature_per_model.data = best_weights['temperature_per_model'].to(device)

        model.original_tuned_model_weights = model.model_weights.data.clone()
        model.original_tuned_class_attention = model.class_attention.data.clone()

    print("✓ Weights applied successfully!")
    print("✓ Original tuned weights saved for test-time blending")

    print("\nStep 7: Computing expected ensemble performance...")

    weighted_preds = np.zeros_like(all_targets)
    for m in range(model.num_models):
        for c in range(model.num_classes):
            weight = class_weights_matrix[m, c]
            weighted_preds[:, c] += weight * all_probs[m, :, c]

    ensemble_aucs = []
    ensemble_accs = []
    ensemble_maps = []
    for c in range(model.num_classes):
        try:
            auc = roc_auc_score(all_targets[:, c], weighted_preds[:, c])
            ensemble_aucs.append(auc)
        except:
            ensemble_aucs.append(0.0)

        binary_preds = (weighted_preds[:, c] > 0.5).astype(int)
        acc = accuracy_score(all_targets[:, c], binary_preds)
        ensemble_accs.append(acc)

        try:
            ap = average_precision_score(all_targets[:, c], weighted_preds[:, c])
            ensemble_maps.append(ap)
        except:
            ensemble_maps.append(0.0)

    avg_ensemble_auc = np.mean([a for a in ensemble_aucs if a > 0])
    avg_ensemble_acc = np.mean(ensemble_accs)
    avg_ensemble_map = np.mean([a for a in ensemble_maps if a > 0])
    avg_best_individual = performance_matrix.max(axis=0).mean()
    avg_best_individual_acc = accuracy_matrix.max(axis=0).mean()
    avg_best_individual_map = map_matrix.max(axis=0).mean()

    print(f"\nValidation set results (AUC):")
    print(f"  Best individual model (avg): {avg_best_individual:.4f}")
    print(f"  Weighted ensemble (expected): {avg_ensemble_auc:.4f}")
    print(f"  Improvement: +{avg_ensemble_auc - avg_best_individual:.4f}")

    print(f"\nValidation set results (Accuracy):")
    print(f"  Best individual model (avg): {avg_best_individual_acc:.4f}")
    print(f"  Weighted ensemble (expected): {avg_ensemble_acc:.4f}")
    print(f"  Improvement: +{avg_ensemble_acc - avg_best_individual_acc:.4f}")

    print(f"\nValidation set results (mAP):")
    print(f"  Best individual model (avg): {avg_best_individual_map:.4f}")
    print(f"  Weighted ensemble (expected): {avg_ensemble_map:.4f}")
    print(f"  Improvement: +{avg_ensemble_map - avg_best_individual_map:.4f}")
    sys.stdout.flush()

    print("\nSaving performance metrics to output directory...")
    np.save(os.path.join(args.output_dir, 'performance_matrix_auc.npy'), performance_matrix)
    np.save(os.path.join(args.output_dir, 'performance_matrix_accuracy.npy'), accuracy_matrix)
    np.save(os.path.join(args.output_dir, 'performance_matrix_map.npy'), map_matrix)

    metrics_summary = {
        'model_names': model.model_names,
        'model_avg_auc': model_avg_aucs.tolist(),
        'model_avg_accuracy': model_avg_accs.tolist(),
        'model_avg_map': model_avg_maps.tolist(),
        'ensemble_avg_auc': float(avg_ensemble_auc),
        'ensemble_avg_accuracy': float(avg_ensemble_acc),
        'ensemble_avg_map': float(avg_ensemble_map),
        'best_individual_auc': float(avg_best_individual),
        'best_individual_accuracy': float(avg_best_individual_acc),
        'best_individual_map': float(avg_best_individual_map)
    }

    with open(os.path.join(args.output_dir, 'performance_metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"✓ Saved AUC matrix to: {os.path.join(args.output_dir, 'performance_matrix_auc.npy')}")
    print(f"✓ Saved Accuracy matrix to: {os.path.join(args.output_dir, 'performance_matrix_accuracy.npy')}")
    print(f"✓ Saved mAP matrix to: {os.path.join(args.output_dir, 'performance_matrix_map.npy')}")
    print(f"✓ Saved metrics summary to: {os.path.join(args.output_dir, 'performance_metrics_summary.json')}")
    sys.stdout.flush()

    print("\n✓ Performance-based weight tuning completed!")
    print("="*60)
    sys.stdout.flush()

    return best_weights


def tune_weights_on_validation_supervised(data_loader, model, device, args, num_epochs=20, lr=0.01):
    """
    Tune ensemble weights on validation set using supervised optimization with labels.

    This method uses validation set labels to optimize ensemble weights by:
    1. Collecting predictions from all models
    2. Using gradient descent to optimize weights that minimize validation loss
    3. Maximizing validation AUC through learnable weight parameters

    Args:
        data_loader: Validation data loader (with labels)
        model: Ensemble model
        device: Device to run on
        args: Arguments
        num_epochs: Number of optimization epochs
        lr: Learning rate for weight optimization

    Returns:
        Best tuned weights
    """
    print("\n" + "="*60)
    print("Tuning Weights on Validation Set (SUPERVISED with labels)")
    print("="*60)
    print(f"Validation samples: {len(data_loader.dataset)}")
    print(f"Optimization epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print()

    model.eval()

    print("Step 1: Collecting predictions from all models...")
    all_individual_probs = []
    all_targets = []

    for batch_idx, batch in enumerate(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            batch_probs = []
            for individual_model in model.models:
                logits = individual_model(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs)

            batch_probs_stack = torch.stack(batch_probs, dim=0)
            all_individual_probs.append(batch_probs_stack)
            all_targets.append(target)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")

    all_probs = torch.cat(all_individual_probs, dim=1)
    all_targets = torch.cat(all_targets, dim=0)
    print(f"✓ Collected predictions: {all_probs.shape}")
    print(f"✓ Collected targets: {all_targets.shape}")

    print("\nStep 2: Optimizing weights with gradient descent...")

    learnable_model_weights = torch.nn.Parameter(model.model_weights.data.clone())
    learnable_class_attention = torch.nn.Parameter(model.class_attention.data.clone())
    learnable_conf_temperature = torch.nn.Parameter(model.confidence_temperature.data.clone())

    optimizer = torch.optim.Adam([
        {'params': learnable_model_weights, 'lr': lr},
        {'params': learnable_class_attention, 'lr': lr * 0.5},
        {'params': learnable_conf_temperature, 'lr': lr * 0.1},
    ])

    criterion = torch.nn.BCELoss()
    best_auc = 0.0
    best_weights = None
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        model_weights_norm = F.softmax(learnable_model_weights, dim=0)
        class_attention_norm = F.softmax(learnable_class_attention, dim=0)

        weights = (model_weights_norm.unsqueeze(1).unsqueeze(2) *
                  class_attention_norm.unsqueeze(1))
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

        ensemble_probs = (all_probs * weights).sum(dim=0)
        ensemble_probs = torch.clamp(ensemble_probs, 1e-7, 1 - 1e-7)

        loss = criterion(ensemble_probs, all_targets)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            learnable_conf_temperature.data = torch.clamp(learnable_conf_temperature.data, 0.5, 5.0)

        with torch.no_grad():
            ensemble_probs_np = ensemble_probs.cpu().numpy()
            targets_np = all_targets.cpu().numpy()

            auc_scores = []
            acc_scores = []
            for c in range(model.num_classes):
                try:
                    auc = roc_auc_score(targets_np[:, c], ensemble_probs_np[:, c])
                    auc_scores.append(auc)
                except:
                    auc_scores.append(0.0)

                binary_preds = (ensemble_probs_np[:, c] > 0.5).astype(int)
                acc = accuracy_score(targets_np[:, c], binary_preds)
                acc_scores.append(acc)

            avg_auc = np.mean([a for a in auc_scores if a > 0])
            avg_acc = np.mean(acc_scores)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{num_epochs}: Loss={loss.item():.4f}, Val AUC={avg_auc:.4f}, Val Acc={avg_acc:.4f}")

            if avg_auc > best_auc:
                best_auc = avg_auc
                best_weights = {
                    'model_weights': learnable_model_weights.data.clone(),
                    'class_attention': learnable_class_attention.data.clone(),
                    'confidence_temperature': learnable_conf_temperature.data.clone(),
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    print(f"\n✓ Best validation AUC: {best_auc:.4f}")

    print("\nStep 3: Analyzing optimized weights...")

    with torch.no_grad():
        model_weights_final = F.softmax(best_weights['model_weights'], dim=0).cpu().numpy()
        print("\nOptimized model-level weights:")
        for i, (name, weight) in enumerate(zip(model.model_names, model_weights_final)):
            print(f"  {name:25s}: {weight:.4f}")

        class_attention_final = F.softmax(best_weights['class_attention'], dim=0).cpu().numpy()
        class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                      'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                      'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                      'Pleural_Thickening', 'Hernia']

        print("\nOptimized class-level specialization (top class per model):")
        for m in range(model.num_models):
            top_classes = np.argsort(class_attention_final[m])[-3:][::-1]
            print(f"  {model.model_names[m]:25s}: ", end="")
            for c_idx in top_classes:
                print(f"{class_names[c_idx]}({class_attention_final[m, c_idx]:.3f}) ", end="")
            print()

        print(f"\nOptimized confidence temperature: {best_weights['confidence_temperature'].item():.4f}")

    print("\nStep 4: Computing per-model temperature calibration...")
    temperature_per_model = torch.ones(model.num_models, device=device)

    for m in range(model.num_models):
        model_probs = all_probs[m].cpu().numpy()
        model_targets = all_targets.cpu().numpy()

        model_aucs = []
        for c in range(model.num_classes):
            try:
                auc = roc_auc_score(model_targets[:, c], model_probs[:, c])
                model_aucs.append(auc)
            except:
                model_aucs.append(0.0)

        avg_auc = np.mean([a for a in model_aucs if a > 0])

        if avg_auc > 0.85:
            T = 1.0
        elif avg_auc > 0.80:
            T = 1.1
        elif avg_auc > 0.75:
            T = 1.2
        else:
            T = 0.9

        temperature_per_model[m] = T
        print(f"  {model.model_names[m]:25s}: T={T:.2f} (Val AUC={avg_auc:.4f})")

    best_weights['temperature_per_model'] = temperature_per_model

    print("\nStep 5: Applying optimized weights to model...")
    with torch.no_grad():
        model.model_weights.data = best_weights['model_weights']
        model.class_attention.data = best_weights['class_attention']
        model.confidence_temperature.data = best_weights['confidence_temperature']
        model.temperature_per_model.data = best_weights['temperature_per_model']

    print("✓ Weight tuning completed!")
    print("="*60)

    return best_weights


@torch.no_grad()
def tune_weights_on_validation(data_loader, model, device, args, num_iterations=5):
    """
    Tune ensemble weights on validation set using unsupervised optimization (DEPRECATED).

    NOTE: This is the old unsupervised version. Use tune_weights_on_validation_supervised() instead
    when validation labels are available for better performance.

    Strategy:
    1. Run inference on validation set to collect statistics
    2. Optimize weights based on multiple unsupervised criteria:
       - Prediction consistency (models agreeing on confident predictions)
       - Prediction sharpness (avoiding ambiguous predictions)
       - Model-specific performance indicators (per-class confidence)
    3. Iteratively refine weights

    Args:
        data_loader: Validation data loader
        model: Ensemble model
        device: Device to run on
        args: Arguments
        num_iterations: Number of refinement iterations

    Returns:
        Best tuned weights
    """
    print("\n" + "="*60)
    print("Tuning Weights on Validation Set (Unsupervised)")
    print("="*60)
    print(f"Validation samples: {len(data_loader.dataset)}")
    print(f"Refinement iterations: {num_iterations}")
    print()

    model.eval()

    all_individual_probs = []
    all_details_list = []

    print("Step 1: Collecting validation set predictions...")
    for batch_idx, batch in enumerate(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            _, details = model(images, return_details=True)

        all_individual_probs.append(details['individual_probs'])
        all_details_list.append(details)

        if (batch_idx + 1) % 20 == 0:
            print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")

    all_probs = torch.cat(all_individual_probs, dim=1)
    print(f"✓ Collected predictions: {all_probs.shape}")

    best_weights = {
        'model_weights': model.model_weights.data.clone(),
        'class_attention': model.class_attention.data.clone(),
        'confidence_temperature': model.confidence_temperature.data.clone(),
    }

    print("\nStep 2: Optimizing weights based on validation statistics...")

    print("\n--- Criterion 1: Model-level Performance ---")
    model_scores = []

    for m in range(model.num_models):
        model_probs = all_probs[m]

        sharpness = torch.abs(model_probs - 0.5).mean().item()

        confidence = torch.max(model_probs, dim=1)[0].mean().item()

        consistency = 1.0 / (model_probs.std().item() + 0.01)

        score = 0.4 * sharpness + 0.4 * confidence + 0.2 * consistency
        model_scores.append(score)

        print(f"  {model.model_names[m]:25s}: "
              f"sharpness={sharpness:.4f}, confidence={confidence:.4f}, "
              f"consistency={consistency:.4f} -> score={score:.4f}")

    model_scores_tensor = torch.tensor(model_scores, device=device)
    model_scores_norm = model_scores_tensor / model_scores_tensor.sum()
    best_weights['model_weights'] = torch.log(model_scores_norm + 1e-8)

    print(f"\n✓ Optimized model weights: {model_scores_norm.cpu().numpy()}")

    print("\n--- Criterion 2: Class-level Specialization ---")
    class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                   'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                   'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                   'Pleural_Thickening', 'Hernia']

    class_scores = torch.zeros(model.num_models, model.num_classes, device=device)

    for m in range(model.num_models):
        for c in range(model.num_classes):
            class_probs = all_probs[m, :, c]

            class_sharpness = torch.abs(class_probs - 0.5).mean().item()
            class_confidence = class_probs.max().item()

            class_scores[m, c] = 0.6 * class_sharpness + 0.4 * class_confidence

    class_scores_norm = class_scores / (class_scores.sum(dim=0, keepdim=True) + 1e-8)
    best_weights['class_attention'] = torch.log(class_scores_norm.t() + 1e-8).t()

    for m in range(model.num_models):
        top_classes = torch.topk(class_scores_norm[m], k=3)
        top_class_names = [class_names[idx] for idx in top_classes.indices.cpu().numpy()]
        top_class_scores = top_classes.values.cpu().numpy()
        print(f"  {model.model_names[m]:25s}: Top classes: ", end="")
        for name, score in zip(top_class_names, top_class_scores):
            print(f"{name}({score:.3f}) ", end="")
        print()

    print("\n--- Criterion 3: Temperature Calibration ---")
    temperature_per_model = torch.zeros(model.num_models, device=device)

    for m in range(model.num_models):
        model_probs = all_probs[m]

        sharpness = torch.abs(model_probs - 0.5).mean().item()

        if sharpness > 0.35:
            T = 1.3
        elif sharpness < 0.20:
            T = 0.8
        else:
            T = 1.0

        temperature_per_model[m] = T
        status = "→ Smoothing" if T > 1.05 else ("→ Sharpening" if T < 0.95 else "→ Keep")
        print(f"  {model.model_names[m]:25s}: T={T:.2f} (sharpness={sharpness:.3f}) {status}")

    model.temperature_per_model.data = temperature_per_model

    print("\n--- Criterion 4: Confidence Temperature ---")

    agreements = []
    confidences = []

    for batch_details in all_details_list:
        conf_weights = batch_details['confidence_weights']
        confidences.append(conf_weights.mean(dim=0))

    avg_confidence = torch.stack(confidences).mean(dim=0)

    conf_std = torch.stack(confidences).std().item()

    if conf_std < 0.1:
        optimal_temp = min(model.confidence_temperature.item() * 1.2, 5.0)
    elif conf_std > 0.3:
        optimal_temp = max(model.confidence_temperature.item() * 0.8, 1.0)
    else:
        optimal_temp = model.confidence_temperature.item()

    best_weights['confidence_temperature'] = torch.tensor(optimal_temp, device=device)
    print(f"  Confidence std: {conf_std:.4f}")
    print(f"  Optimal temperature: {optimal_temp:.2f}")

    print("\nStep 3: Applying optimized weights...")
    with torch.no_grad():
        model.model_weights.data = best_weights['model_weights']
        model.class_attention.data = best_weights['class_attention']
        model.confidence_temperature.data = best_weights['confidence_temperature']

    print("✓ Weight tuning completed!")
    print("="*60)

    return best_weights


@torch.no_grad()
def evaluate_ensemble(data_loader, model, device, args, mode='test', use_tuned_weights=True):
    """
    Evaluate ensemble model with dynamic confidence-based weight adjustment.

    Args:
        data_loader: Data loader
        model: Ensemble model
        device: Device
        args: Arguments
        mode: 'val' or 'test'
        use_tuned_weights: If True, combine tuned weights with adaptive weights
    """
    criterion = torch.nn.BCEWithLogitsLoss()

    model.eval()

    all_outputs = []
    all_targets = []
    all_details = []

    total_loss = 0.0
    num_batches = 0

    accumulated_model_confidences = []
    accumulated_class_confidences = []

    print(f"Evaluating ensemble model on {mode} set...")
    if use_tuned_weights and hasattr(model, 'tuned_weights_path') and model.tuned_weights_path:
        print("✓ Using validation-tuned weights combined with adaptive weighting")
    else:
        print("Using dynamic confidence weighting...")
    print("Accumulating confidence statistics from ALL batches for base_weights")

    for batch_idx, batch in enumerate(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output, details = model(images, return_details=True)
            loss = criterion(output, target)

        individual_probs = details['individual_probs']

        batch_model_confidences = []
        batch_class_confidences = []

        for m in range(model.num_models):
            model_probs = individual_probs[m]

            model_conf = torch.max(model_probs, dim=1)[0].mean().item()
            batch_model_confidences.append(model_conf)

            class_conf = []
            for c in range(model.num_classes):
                class_probs = model_probs[:, c]
                class_confidence = torch.mean(torch.abs(class_probs - 0.5)).item()
                class_conf.append(class_confidence)

            batch_class_confidences.append(class_conf)

        accumulated_model_confidences.append(batch_model_confidences)
        accumulated_class_confidences.append(batch_class_confidences)

        if (batch_idx + 1) % 100 == 0:
            avg_model_confidences = np.mean(accumulated_model_confidences, axis=0).tolist()
            avg_class_confidences = np.mean(accumulated_class_confidences, axis=0).tolist()

            with torch.no_grad():
                if use_tuned_weights:
                    class_confidences_tensor = torch.tensor(avg_class_confidences, device=device)
                    class_confidences_norm = class_confidences_tensor / (class_confidences_tensor.sum(dim=0, keepdim=True) + 1e-8)

                    ori_class_attention = F.softmax(model.original_tuned_class_attention, dim=0)
                    current_class_attention = F.softmax(model.class_attention, dim=0)
                    alpha_factor = 0.1
                    beta_factor = 0.1

                    blended_class_attention = (1 - alpha_factor - beta_factor) * ori_class_attention + \
                                             beta_factor * class_confidences_norm + \
                                             alpha_factor * current_class_attention

                    blended_class_attention = blended_class_attention / (blended_class_attention.sum(dim=0, keepdim=True) + 1e-8)
                    model.class_attention.data = torch.log(blended_class_attention.t() + 1e-8).t()

                    if (batch_idx + 1) == 100:
                        print(f"\n✓ Using validation-tuned model weights (FIXED):")
                        print(f"  Model weights: {F.softmax(model.model_weights, dim=0).cpu().numpy()}")
                        print(f"\n✓ Adjusting class attention with alpha_factor={alpha_factor} and beta_factor={beta_factor}:")
                        print(f"  ({alpha_factor} tuned weights + {beta_factor} dynamic adjustment + {1 - alpha_factor - beta_factor} original)")
                        print(f"  Current class attention: {current_class_attention.cpu().numpy()}")
                        print(f"  Dynamic class attention: {class_confidences_norm.cpu().numpy()}")

                else:
                    model_confidences_tensor = torch.tensor(avg_model_confidences, device=device)
                    model_confidences_tensor = model_confidences_tensor / model_confidences_tensor.sum()
                    model.model_weights.data = torch.log(model_confidences_tensor + 1e-8)

                    class_confidences_tensor = torch.tensor(avg_class_confidences, device=device)
                    class_confidences_norm = class_confidences_tensor / (class_confidences_tensor.sum(dim=0, keepdim=True) + 1e-8)
                    model.class_attention.data = torch.log(class_confidences_norm.t() + 1e-8).t()

                    if (batch_idx + 1) == 100:
                        print(f"\n✓ Updated weights dynamically after {batch_idx + 1} batches:")
                        print(f"  Model confidences: {avg_model_confidences}")
                        print(f"  Model weights: {model_confidences_tensor.cpu().numpy()}")
                        print()

        total_loss += loss.item()
        num_batches += 1

        all_outputs.append(output)
        all_targets.append(target)
        all_details.append(details)

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

    outputs = torch.cat(all_outputs, dim=0).sigmoid().cpu().numpy()
    targets = torch.cat(all_targets, dim=0).cpu().numpy()

    avg_loss = total_loss / num_batches

    num_classes = args.nb_classes
    auc_each_class = []
    acc_each_class = []
    map_each_class = []

    for i in range(num_classes):
        try:
            auc = roc_auc_score(targets[:, i], outputs[:, i])
            auc_each_class.append(auc)
        except:
            auc_each_class.append(0.0)

        binary_preds = (outputs[:, i] > 0.5).astype(int)
        acc = accuracy_score(targets[:, i], binary_preds)
        acc_each_class.append(acc)

        try:
            ap = average_precision_score(targets[:, i], outputs[:, i])
            map_each_class.append(ap)
        except:
            map_each_class.append(0.0)

    print("\nPer-class AUC scores:")
    class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                   'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                   'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                   'Pleural_Thickening', 'Hernia']

    for i, (name, auc) in enumerate(zip(class_names, auc_each_class)):
        print(f"  {name:20s}: {auc:.4f}")

    auc_each_class_array = np.array(auc_each_class)
    auc_avg = np.average(auc_each_class_array[auc_each_class_array != 0])

    print(f"\nAverage AUC: {auc_avg:.4f}")

    print("\nPer-class Accuracy scores:")
    for i, (name, acc) in enumerate(zip(class_names, acc_each_class)):
        print(f"  {name:20s}: {acc:.4f}")

    acc_avg = np.mean(acc_each_class)
    print(f"\nAverage Accuracy: {acc_avg:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")

    print("\nPer-class mAP (Average Precision) scores:")
    for i, (name, map_val) in enumerate(zip(class_names, map_each_class)):
        print(f"  {name:20s}: {map_val:.4f}")

    map_each_class_array = np.array(map_each_class)
    map_avg = np.average(map_each_class_array[map_each_class_array != 0])
    print(f"\nAverage mAP: {map_avg:.4f}")
    sys.stdout.flush()

    print("\n" + "="*60)
    print("Ensemble Weight Analysis")
    print("="*60)

    model_weights = model.model_weights.softmax(0).cpu().numpy()
    print("\nModel-level weights:")
    for i, (name, weight) in enumerate(zip(model.model_names, model_weights)):
        print(f"  {name:20s}: {weight:.4f}")

    if model.enable_temperature_scaling:
        print("\nUnsupervised Temperature Scaling:")
        temps = model.temperature_per_model.cpu().numpy()
        for i, (name, temp) in enumerate(zip(model.model_names, temps)):
            status = "Smooth" if temp > 1.1 else ("Sharpen" if temp < 0.9 else "Normal")
            print(f"  {name:20s}: T={temp:.2f} ({status})")

    class_attention = model.class_attention.softmax(0).cpu().numpy()
    print("\nClass-level attention (top class per model):")
    for i, name in enumerate(model.model_names):
        top_class_idx = np.argmax(class_attention[i])
        top_class_weight = class_attention[i, top_class_idx]
        print(f"  {name:20s} -> {class_names[top_class_idx]:20s}: {top_class_weight:.4f}")

    if model.enable_uncertainty_weighting or model.enable_agreement_weighting:
        print("\nUnsupervised Adaptive Weighting:")
        print("  ✓ Agreement-based weighting enabled")
        print("  ✓ Uncertainty-based weighting enabled")
        print("  Strategy: High agreement -> equal weights | High disagreement -> trust confident models")

    np.save(os.path.join(args.output_dir, 'ensemble_y_gt.npy'), targets)
    np.save(os.path.join(args.output_dir, 'ensemble_y_pred.npy'), outputs)

    return {
        'loss': avg_loss,
        'auc_avg': auc_avg,
        'auc_each_class': auc_each_class,
        'acc_avg': acc_avg,
        'acc_each_class': acc_each_class,
        'map_avg': map_avg,
        'map_each_class': map_each_class
    }


@torch.no_grad()
def evaluate_individual_models(data_loader, model, device, args):
    """Evaluate individual models and compute oracle performance"""
    model.eval()

    all_targets = []
    all_predictions = []

    print("Evaluating individual models...")
    for batch_idx, batch in enumerate(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_predictions = []
        with torch.cuda.amp.autocast():
            for individual_model in model.models:
                output = individual_model(images)
                probs = torch.sigmoid(output)
                batch_predictions.append(probs)

        all_predictions.append(torch.stack(batch_predictions, dim=0))
        all_targets.append(target)

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

    predictions = torch.cat(all_predictions, dim=1).cpu().numpy()
    targets = torch.cat(all_targets, dim=0).cpu().numpy()

    num_classes = args.nb_classes

    print("\n" + "="*60)
    print("Individual Model Performance")
    print("="*60)

    class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                   'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                   'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                   'Pleural_Thickening', 'Hernia']

    individual_results = {}

    for i, model_name in enumerate(model.model_names):
        preds = predictions[i]

        auc_each_class = []
        acc_each_class = []
        map_each_class = []
        for c in range(num_classes):
            try:
                auc = roc_auc_score(targets[:, c], preds[:, c])
                auc_each_class.append(auc)
            except:
                auc_each_class.append(0.0)

            binary_preds = (preds[:, c] > 0.5).astype(int)
            acc = accuracy_score(targets[:, c], binary_preds)
            acc_each_class.append(acc)

            try:
                ap = average_precision_score(targets[:, c], preds[:, c])
                map_each_class.append(ap)
            except:
                map_each_class.append(0.0)

        auc_avg = np.mean([a for a in auc_each_class if a > 0])
        acc_avg = np.mean(acc_each_class)
        map_avg = np.mean([m for m in map_each_class if m > 0])

        print(f"\n{model_name}:")
        print(f"  Average AUC: {auc_avg:.4f}")
        print(f"  Average Accuracy: {acc_avg:.4f}")
        print(f"  Average mAP: {map_avg:.4f}")

        print(f"  Per-class mAP:")
        for c, (name, map_val) in enumerate(zip(class_names, map_each_class)):
            print(f"    {name:20s}: {map_val:.4f}")

        individual_results[model_name] = {
            'auc_avg': float(auc_avg),
            'auc_each_class': [float(x) for x in auc_each_class],
            'acc_avg': float(acc_avg),
            'acc_each_class': [float(x) for x in acc_each_class],
            'map_avg': float(map_avg),
            'map_each_class': [float(x) for x in map_each_class]
        }

        safe_model_name = model_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')
        model_results_path = os.path.join(args.output_dir, f'{safe_model_name}_results.json')
        with open(model_results_path, 'w') as f:
            json.dump(individual_results[model_name], f, indent=4)
        print(f"  Results saved to: {model_results_path}")

    print("\n" + "="*60)
    print("Oracle Performance (Union of All Models)")
    print("="*60)
    print("Oracle: At least ONE model correct = correct prediction")
    print("  - If ANY model predicts >0.5 AND label=1 → correct")
    print("  - If ANY model predicts <0.5 AND label=0 → correct")

    oracle_auc = []
    oracle_details = []

    for c in range(num_classes):
        class_preds = predictions[:, :, c]
        class_targets = targets[:, c]

        oracle_preds = np.zeros(len(class_targets))

        for i in range(len(class_targets)):
            sample_preds = class_preds[:, i]
            target = class_targets[i]

            if target == 1:
                oracle_preds[i] = np.max(sample_preds)
            else:
                oracle_preds[i] = np.min(sample_preds)

        try:
            auc = roc_auc_score(class_targets, oracle_preds)
            oracle_auc.append(auc)

            individual_aucs = []
            for m in range(len(model.model_names)):
                try:
                    individual_aucs.append(roc_auc_score(class_targets, class_preds[m]))
                except:
                    individual_aucs.append(0.0)

            oracle_details.append({
                'class': c,
                'auc': auc,
                'individual_aucs': individual_aucs
            })
        except:
            oracle_auc.append(0.0)
            oracle_details.append(None)

    oracle_avg = np.mean([a for a in oracle_auc if a > 0])
    print(f"\nOracle Average AUC: {oracle_avg:.4f}")
    print("(Perfect model selection for each sample)")

    print("\nPer-class Oracle vs Best Individual:")
    class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                   'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                   'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                   'Pleural_Thickening', 'Hernia']

    for c, name in enumerate(class_names):
        if oracle_details[c] is not None:
            best_individual = max(oracle_details[c]['individual_aucs'])
            oracle_val = oracle_details[c]['auc']
            gap = oracle_val - best_individual
            print(f"  {name:20s}: Oracle {oracle_val:.4f} | Best {best_individual:.4f} | Gap {gap:+.4f}")

    return oracle_avg, individual_results


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Ensemble Model for Biomedical Image Analysis',
        add_help=False
    )

    parser.add_argument('--eva_x_checkpoint', type=str,
                        required=True,
                        help='Path to EVA-X-Tiny checkpoint')
    parser.add_argument('--mgca_checkpoint', type=str, required=True,
                        help='Path to MGCA ResNet50 checkpoint')
    parser.add_argument('--medical_mae_checkpoint', type=str, required=True,
                        help='Path to Medical MAE DenseNet121 checkpoint')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--test_list', type=str, required=True,
                        help='Path to test list file')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--nb_classes', default=14, type=int)
    parser.add_argument('--num_workers', default=10, type=int)

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save outputs')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='chestxray', type=str)

    parser.add_argument('--evaluate_individual', action='store_true',
                        help='Also evaluate individual models and oracle')
    parser.add_argument('--tune_on_val', action='store_true',
                        help='Tune weights on validation set before testing')
    parser.add_argument('--tuned_weights_path', type=str, default=None,
                        help='Path to pre-tuned weights (if available)')
    parser.add_argument('--save_tuned_weights', action='store_true',
                        help='Save tuned weights after validation tuning')
    parser.add_argument('--tune_epochs', type=int, default=20,
                        help='Number of epochs for supervised weight tuning')
    parser.add_argument('--tune_lr', type=float, default=0.01,
                        help='Learning rate for supervised weight tuning')

    parser.add_argument('--train_list', default=None, type=str)
    parser.add_argument('--val_list', default=None, type=str)
    parser.add_argument('--build_timm_transform', action='store_true', default=True)
    parser.add_argument('--aa', default='rand-m6-mstd0.5-inc1', type=str)
    parser.add_argument('--pin_mem', action='store_true', default=True)

    parser.add_argument('--model', default='eva02_base_patch16', type=str,
                        help='Model name for transform normalization')
    parser.add_argument('--color_jitter', type=float, default=None)
    parser.add_argument('--reprob', type=float, default=0.0)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--data_pct', type=float, default=1.0,
                        help='Percentage of data to use (for debugging)')

    return parser


def main(args):
    print('='*60)
    print('No Training Needed: Ensemble Experts for')
    print('Universal Biomedical Image Analysis')
    print('='*60)
    print(f"\nOutput directory: {args.output_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f'train_ensemble_{timestamp}.log')
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)

    print(f"Logging to: {log_file}")
    print(f"Start time: {timestamp}")

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    data_loader_val = None
    if args.tune_on_val or args.val_list:
        print(f"\nLoading validation dataset from {args.data_path}")
        dataset_val = build_dataset_chest_xray(split='val', args=args)
        print(f"Validation dataset size: {len(dataset_val)}")

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    print(f"\nLoading test dataset from {args.data_path}")
    dataset_test = build_dataset_chest_xray(split='test', args=args)
    print(f"Test dataset size: {len(dataset_test)}")

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    global model_configs
    model_configs = [
        {
            'name': 'EVA-X-Tiny (EVA02-Tiny/16)',
            'type': 'vit',
            'model_name': 'eva02_tiny_patch16_xattn_fusedLN_SwiGLU_preln_RoPE',
            'checkpoint': args.eva_x_checkpoint,
            'num_classes': args.nb_classes,
            'input_size': args.input_size,
        },
        {
            'name': 'MGCA (ResNet50)',
            'type': 'resnet50',
            'checkpoint': args.mgca_checkpoint,
            'num_classes': args.nb_classes,
        },
        {
            'name': 'Medical MAE (DenseNet121)',
            'type': 'densenet121',
            'checkpoint': args.medical_mae_checkpoint,
            'num_classes': args.nb_classes,
        },
    ]

    print("\n" + "="*60)
    print("Building Ensemble Model")
    print("="*60)

    model = EnsembleExpertModel(
        model_configs,
        num_classes=args.nb_classes,
        enable_feature_fusion=False,
        tuned_weights_path=args.tuned_weights_path
    )
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M")

    print("\nNote: All model weights are frozen - no training needed!")
    print("Only ensemble combination weights are learnable.\n")

    if args.tune_on_val and data_loader_val is not None:
        print("\n" + "="*60)
        print("Step 1: Tuning Weights on Validation Set")
        print("="*60)

        tuned_weights = tune_weights_on_validation_performance_based(
            data_loader_val, model, device, args
        )

        if args.save_tuned_weights:
            tuned_weights_save_path = os.path.join(args.output_dir, 'tuned_weights.pth')
            model.save_tuned_weights(tuned_weights_save_path)
            print(f"\n✓ Saved tuned weights to: {tuned_weights_save_path}")

        print("\n" + "="*60)
        print("Validation Set Performance (with tuned weights)")
        print("="*60)
        val_stats = evaluate_ensemble(data_loader_val, model, device, args, mode='val', use_tuned_weights=True)
        print(f"Validation AUC: {val_stats['auc_avg']:.4f}")
        print(f"Validation Accuracy: {val_stats['acc_avg']:.4f}")
        print(f"Validation mAP: {val_stats['map_avg']:.4f}")

    if args.evaluate_individual:
        oracle_auc, individual_results = evaluate_individual_models(data_loader_test, model, device, args)
    else:
        individual_results = {}

    print("\n" + "="*60)
    print("Step 2: Evaluating Ensemble Model on Test Set")
    print("="*60)

    start_time = time.time()
    test_stats = evaluate_ensemble(data_loader_test, model, device, args, mode='test',
                                   use_tuned_weights=(args.tuned_weights_path is not None or args.tune_on_val))
    eval_time = time.time() - start_time

    print(f"\nEvaluation completed in {eval_time:.2f} seconds")

    results = {
        'ensemble_auc_avg': test_stats['auc_avg'],
        'ensemble_auc_each_class': test_stats['auc_each_class'],
        'ensemble_acc_avg': test_stats['acc_avg'],
        'ensemble_acc_each_class': test_stats['acc_each_class'],
        'ensemble_map_avg': test_stats['map_avg'],
        'ensemble_map_each_class': test_stats['map_each_class'],
        'ensemble_loss': test_stats['loss'],
        'evaluation_time': eval_time,
    }

    if args.evaluate_individual:
        results['oracle_auc'] = oracle_auc
        results['individual_models'] = individual_results

    with open(os.path.join(args.output_dir, 'ensemble_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Results saved to {args.output_dir}")

    if args.evaluate_individual:
        print(f"\nEnsemble AUC: {test_stats['auc_avg']:.4f}")
        print(f"Ensemble Accuracy: {test_stats['acc_avg']:.4f}")
        print(f"Ensemble mAP: {test_stats['map_avg']:.4f}")
        print(f"Oracle AUC:   {oracle_auc:.4f}")
        print(f"Gap (AUC):    {oracle_auc - test_stats['auc_avg']:.4f}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
