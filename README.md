# CT-PneumoSeg
AI model to detect pneumothorax in CT scans and generate segmentation masks highlighting the pathological regions.

# Medical Image Lesion Detection - Cascade Pipeline

## 🎯 Strategy Overview

This project implements a **two-stage cascade pipeline** for pneumothorax detection in chest X-rays. Instead of running a heavy segmentation model on every image, we use a lightweight classifier as a filter, significantly improving inference speed while maintaining high detection accuracy.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CASCADE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input Image ──► [CLASSIFIER] ──► Suspect? ──YES──► [SEGMENTER] ──► Mask│
│                        │                                                │
│                        └────────► NO ──► Empty Mask (-1)                │
│                                                                         │
│           n    ~77% of images are healthy → Skip segmentation           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🏗️ Architecture

### Stage 1: Classifier (The Filter)

**Model:** EfficientNet-B3  
**Role:** Binary classification - "Does this image contain a lesion?"  
**Priority:** **High Sensitivity (Recall)** - Never miss a lesion

| Component | Details |
|-----------|---------|
| Backbone | EfficientNet-B3 (pretrained ImageNet) |
| Input | Grayscale 512×512 |
| Output | Probability score [0, 1] |
| Loss | Focal Loss (α=0.75, γ=2.0) |
| Optimization | F2-Score (Recall weighted 2× more than Precision) |
| Training | Weighted Random Sampler for balanced batches |

**Why EfficientNet-B3?**
- Excellent accuracy/speed tradeoff (~12M parameters)
- Compound scaling provides better feature extraction
- Pretrained weights transfer well to medical imaging

**Threshold Calibration:**  
The classifier threshold is calibrated to achieve **~95% recall** - we accept some false positives because:
1. Missing a lesion (False Negative) is clinically dangerous
2. False positives are filtered by the segmenter in Stage 2
3. Most healthy images are still correctly filtered out

### Stage 2: Segmenter (The Detector)

**Model:** U-Net with ConvNeXt-Tiny Backbone  
**Role:** Pixel-wise segmentation - "Where exactly is the lesion?"  
**Priority:** **Precise localization** with minimal false positives

| Component | Details |
|-----------|---------|
| Encoder | ConvNeXt-Tiny (pretrained ImageNet) |
| Decoder | Residual ConvBlocks with skip connections |
| Input | Grayscale 512×512 |
| Output | Segmentation mask 512×512 |
| Loss | Combo Loss (BCE + Batch Dice) |
| Activation | GELU |

**Why ConvNeXt-Tiny?**
- Modern architecture outperforming traditional ResNets
- Efficient 4× downsampling in stem layer
- Better gradient flow with LayerNorm and GELU
- Strong pretrained representations

**Batch Dice Loss:**  
Instead of computing Dice per-image (which gives perfect 1.0 score on empty masks), we compute Dice across the entire batch. This prevents the model from learning to predict empty masks on healthy images.

```python
# Traditional Dice: Empty prediction on empty mask = 1.0 (perfect but meaningless)
# Batch Dice: Computes intersection/union across ALL images in batch
```

## 📊 Training Strategy

### Classifier Training
```
Dataset: Full dataset with real class distribution (~23% positive)
Sampling: WeightedRandomSampler → 50/50 balanced batches
Validation: Real proportions (for proper threshold calibration)
Metric: F2-Score (β=2 prioritizes recall)
```

### Segmenter Training
```
Dataset: Enriched with lesion cases (configurable ratio, e.g., 75% lesions)
Augmentation: Flip, Rotate, ShiftScale
Validation: Lesion Dice score (only on positive cases)
Metric: Dice coefficient + Classification accuracy
```

## ⚡ Inference Pipeline

```python
for image in dataset:
    # Stage 1: Fast classification
    prob = classifier(image)
    
    if prob < threshold:
        # ~77% of cases: Skip expensive segmentation
        mask = empty_mask()
    else:
        # ~23% of cases: Run segmentation
        mask = segmenter(image)
        
        # Post-processing: Remove small components
        mask = filter_small_components(mask, min_pixels=200)
    
    # Encode result
    rle = encode_mask(mask)
```


## 📁 Output Formats

The pipeline generates two prediction formats:

| File | Description | Use Case |
|------|-------------|----------|
| `predictions_aggregated.csv` | 1 row per image, all lesions merged | Standard submission format |
| `predictions_split.csv` | N rows per image, 1 per lesion | Multi-instance evaluation |

## 🔧 Configuration

All thresholds are configurable in `PipelineConfig`:

```python
class PipelineConfig:
    # Classifier threshold (calibrated for high recall)
    CLASSIFIER_THRESHOLD = 0.35  # Lower = more sensitive
    
    # Segmenter probability threshold
    SEGMENTER_THRESHOLD = 0.5
    
    # Minimum lesion size (pixels)
    MIN_PIXELS = 200  # Filter noise/artifacts
```

## 🗂️ Project Structure

```
├── classifier_efficientnet_b3.py   # Stage 1: Binary classifier
├── segmenter_convnext_tiny.py      # Stage 2: U-Net segmenter
├── pipeline_inference.py           # Combined inference pipeline
├── visualizations/                 # Training curves, confusion matrices
└── outputs/                        # Predictions CSV files
```

- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
