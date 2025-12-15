# Retinal OCT Disease Classification: Custom CNN vs Transfer Learning

A comparative study building a CNN from scratch versus using pretrained ResNet18 for classifying retinal OCT images from the OCTDL dataset.

## Dataset

**OCTDL** - Optical Coherence Tomography images for retinal disease classification
- 7 classes: AMD, DME, ERM, NO (normal), RAO, RVO, VID
- ~2,000 grayscale OCT B-scan images
- Class imbalanced (AMD/NO are majority classes)
- 80/20 train/val split

Source: [OCTDL on Kaggle](https://www.kaggle.com/datasets/shakilrana/octdl-retinal-oct-images-dataset)

## Models

### Custom CNN (From Scratch)

**Architecture:**
- 4 conv blocks: 16→32→64→128 channels
- Each block: Conv3x3 → BatchNorm → ReLU → MaxPool
- Global average pooling + dropout (0.6)
- ~150K parameters

**Training:**
- AdamW optimizer (lr=1e-3, weight_decay=1e-4)
- Class-weighted CrossEntropyLoss for imbalance
- Data augmentation: horizontal flip, rotation ±10°
- 25 epochs with best validation epoch selection

### ResNet18 (Transfer Learning)

**Architecture:**
- ResNet18 pretrained on ImageNet
- Fine-tuned with 7-class output layer
- ~11M parameters

**Training:**
- AdamW optimizer (lr=1e-4, weight_decay=1e-4)
- Standard CrossEntropyLoss
- Same augmentation as custom CNN
- 10 epochs

## Results

| Model | Val Accuracy | Notes |
|-------|--------------|-------|
| Custom CNN (no weighting) | 76.5% | Strong majority-class bias |
| Custom CNN (weighted CE) | **80.4%** | Improved minority-class recall |
| ResNet18 (pretrained) | **93.0%** | Strong baseline with ImageNet features |

## Key Findings

**Custom CNN:**
- Reaches 80% with proper class balancing (vs 14% random baseline)
- Confuses visually similar diseases (AMD ↔ VID, NO ↔ others)
- Good for lightweight deployment but limited by small dataset

**ResNet18:**
- 13% accuracy advantage from pretrained features
- Better rare-class performance (RAO, VID)
- Much faster convergence (10 vs 25 epochs)

**Class Weighting Impact:**
- +4% absolute accuracy improvement
- Reduced "predict NO for everything" behavior
- Essential for medical datasets with imbalance

## Confusion Matrix Insights

**Custom CNN:** Over-predicts majority classes; struggles with AMD/VID distinction due to similar texture patterns.

**ResNet18:** Clean diagonals; minimal inter-class confusion thanks to hierarchical feature learning from ImageNet.
