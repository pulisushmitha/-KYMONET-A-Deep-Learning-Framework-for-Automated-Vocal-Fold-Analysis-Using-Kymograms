## Tertiary Classification (Healthy vs Functional vs Organic)

### Objective
Classify kymogram images into three clinically significant categories:
- **Healthy**
- **Functional Disorders**
- **Organic Disorders**

### Models Evaluated
The following deep learning architectures were trained and compared:
- AlexNet (with ResNet-152 backbone)
- ConvNeXt V2
- DenseNet121
- InceptionV3
- ResNet50V2-MLP
- EfficientNetV2

### Training Setup
- Dataset: Segmented and augmented kymogram images
- Class Distribution After Augmentation:
  - Healthy — 6,219
  - Functional — 2,104
  - Organic — 1,573
- Oversampling: Applied **only to the training split** for class balance
- Epochs: **30**
- Loss Function: **Cross-Entropy / Weighted Loss**
- Batch Size: **32**
- Performance Metrics: Accuracy, AUC, Precision, Recall, Sensitivity, Specificity, F1-Score

### Class-Wise Highlights (InceptionV3)
- **F1-Score (Healthy):** 99.84%
- **F1-Score (Functional):** 99.84%
- **F1-Score (Organic):** 98.50%
- **No major confusion detected in predictions across categories**

### Why Tertiary Classification Is Harder
- Functional and Organic disorders may show similar vibration patterns in early stages
- Boundary cases require deeper feature learning
- Kymograms contain subtle differences in symmetry, mucosal wave, and subglottal pressure patterns

**InceptionV3 overcame this challenge due to:**
- Multiple receptive field sizes (1×1, 3×3, 5×5)
- Auxiliary classifiers improving deep gradient propagation
- Global Average Pooling reducing overfitting

### Takeaway
**Tertiary classification using kymograms and transfer learning architectures is highly reliable for clinical screening of voice disorders.**  
**InceptionV3 is the best-suited model for deployment due to consistent high performance across all three classes.**
