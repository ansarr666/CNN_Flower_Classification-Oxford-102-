# 🌸 Flower Classification using Custom CNN (Oxford 102)

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) built from scratch to classify 102 different species of flowers using the Oxford 102 Flowers dataset.

The project demonstrates end-to-end deep learning workflow including dataset handling, architecture design, training optimization, and performance evaluation — without transfer learning.

---

## 🚀 Project Highlights

- **Architecture:** Custom 5-layer Deep CNN
- **Techniques Used:** Batch Normalization, Dropout, Adaptive Pooling
- **Training Accuracy:** 74.99%
- **Validation Accuracy:** 66.08%
- **Hardware:** NVIDIA GTX 1650 (CUDA Accelerated Training)
- **Training Strategy:** Split-Swap approach (trained on larger split of 6,149 images)

---

## 📊 Dataset Overview

**Dataset:** Oxford 102 Flowers

- 102 flower categories
- 40–258 images per class
- High intra-class variation
- Fine-grained classification challenge

---

## 🛠️ Model Architecture

The CNN was designed to handle high intra-class variance and low inter-class variance.

### Architecture Summary

| Layer | Type | Configuration |
|-------|------|--------------|
| Input | Image | 128 × 128 RGB |
| Conv Block ×5 | Conv2d → BatchNorm → ReLU → MaxPool | Progressive channel expansion |
| Bottleneck | AdaptiveAvgPool2d | Output size 1 × 1 |
| Classifier | Fully Connected | 512 Hidden Units + Dropout(0.5) |
| Output | Linear | 102 Class Logits |

---

## 📈 Training Results

After 20 epochs:

- **Final Training Loss:** 0.872
- **Final Training Accuracy:** 74.99%
- **Final Validation Accuracy:** 66.08%

### Observations

- Mild overfitting observed (train > validation)
- Controlled using:
  - Dropout (0.5)
  - Data Augmentation (Rotation + Horizontal Flip)
  - Normalization

---

## ⚙️ Requirements

- Python 3.10+
- PyTorch (CUDA enabled recommended)
- Torchvision
- Matplotlib
- NumPy

### Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib numpy
