## Binary Classification (Healthy vs Unhealthy)

### Objective
Classify kymogram images into:
- **Healthy**
- **Unhealthy (Functional + Organic combined)**

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
- Split: 70% Train — 15% Validation — 15% Test
- Oversampling: Applied ONLY to the training set to balance classes
- Epochs: **30**
- Loss Function: **Cross-Entropy / Weighted Cross-Entropy**
- Batch Size: **32**
- Evaluation: Accuracy, AUC, Precision, Recall, Sensitivity, Specificity

### Metric Highlights
- **InceptionV3 achieved zero misclassifications** on the binary test set  
- ConvNeXt V2 and AlexNet also delivered **AUC values above 99%**
- All models demonstrated strong separability between Healthy and Unhealthy classes

### Why Binary Classification Performed So Well
- Kymograms provide highly discriminative visual representation of vocal fold vibrations  
- Deep CNN feature extractors (especially Inception modules) capture:
  - Symmetry vs asymmetry
  - Closure patterns
  - Vibration regularity
  - Mucosal wave patterns
- Oversampling + augmentation increased robustness to patient-specific variation

### Takeaway
**Binary classification using optimized deep learning pipelines is highly reliable for detecting vocal fold disorders from kymograms.**  
**InceptionV3 is the recommended architecture for deployment with near-perfect performance.**
