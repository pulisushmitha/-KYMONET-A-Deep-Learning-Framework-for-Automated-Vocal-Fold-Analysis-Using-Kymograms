# -KYMONET-A-Deep-Learning-Framework-for-Automated-Vocal-Fold-Analysis-Using-Kymograms
Generated kymograms using MATLAB, preprocessed vocal fold videos with RIFE and FFmpeg, and built deep learning models for ternary classification. Achieved 98% accuracy through oversampling, hyperparameter tuning, and thorough metric-based evaluation.
# 1. Data Acquisition

The dataset used in this project is the  **BAGLS (Benchmark for Automatic Glottis Segmentation) dataset**, which contains **640 High-Speed Videoendoscopy (HSV) recordings** collected from seven medical institutions across the USA and Europe. These videos include a mixture of **healthy, organic, and functional** voice disorder cases, along with some samples of unspecified health status. 

modified_voice_disoder_deep_lea…

Each recording captures the rapid vibration of the vocal folds at thousands of frames per second, providing extremely fine temporal resolution. This makes HSV essential for understanding the dynamic behavior of the vocal folds during phonation. Out of the 640 videos, **50 samples were discarded** due to incomplete metadata, and the remaining samples were used for kymogram generation and classification.

The dataset includes metadata specifying the type of disorder, allowing the classification structure to be split into:

-**Healthy**

-**Functional disorders** (e.g., muscle tension dysphonia)

-**Organic disorders** (e.g., nodules, cysts, edema, paralysis)

A detailed breakdown of disorder categories is provided in Table 1 of the report. 


## **2. Kymogram Generation Using MATLAB**
A **kymogram** is a 2D visualization that compresses an entire video into a single vertical stack of pixel rows, representing vocal fold vibration over time. It is widely used for detecting asymmetry, irregular closures, and abnormal vibratory cycles.

**Kymogram Generation Pipeline:**

**1**. Load an HSV video into MATLAB.

**2**. Extract the first frame and manually select the **Region of Interest (ROI)** centered around the glottis.

**3**. Confirm ROI boundaries.

**4**. From every frame in the video:

   -A **horizontal scanline** is extracted across the glottis.
   -This line is placed as one row in the output kymogram.

**5**. The resulting image is contrast-enhanced and saved as a .png.

This process was executed for all available videos, and high-quality kymograms were generated for 590+ samples. 

#3. Video Enhancement using RIFE and FFmpeg

Some HSV videos (29 samples) originally produced **noisy or blurred kymograms** due to issues such as motion blur, excessive brightness, or low frame uniformity.

To correct this, two enhancement methods were used:

## **3.1 RIFE (Real-Time Intermediate Flow Estimation)**

RIFE is a deep learning–based frame interpolation model that generates intermediate frames between existing ones.
Benefits:
-Smoothens transitions
-Reduces motion artifacts
-Produces clearer vibratory sequences

After RIFE enhancement, videos were more stable and resulted in improved kymograms. However, only **5 out of 29 videos** were successfully stabilized; the remaining suffered from extreme noise and were discarded based on clinical research practices. 

## **3.2 FFmpeg Slow-Motion Processing**

Videos were slowed down to **0.2× speed** using FFmpeg to increase temporal resolution and allow more accurate scanline extraction. This preprocessing helped reduce inconsistencies caused by rapid oscillations of vocal folds. 

## **4. Kymogram Segmentation and Augmentation**
Original kymograms varied in height (number of frames). To convert them into meaningful input samples:

**Segmentation Procedure:**
-Each kymogram was split into smaller 58 × 121 pixel slices.
-The slicing covered the full height of the kymogram, producing multiple segment images per sample.
-Each segment was given a sequential filename.

**Rotation Augmentation:**
-Each segment was rotated 90° clockwise, effectively converting the time-axis orientation and generating more samples.
This significantly expanded the dataset:
Class	Original Count	After Segmentation & Rotation
| Class       | Original Count | After Segmentation & Augmentation |
|-------------|----------------|-----------------------------------|
| Healthy     | 356            | 6,219                             |
| Functional  | 121            | 2,104                             |
| Organic     | 83             | 1,573                             |


Total dataset size after augmentation: **9,896 images.**
This drastically improved model generalization and reduced overfitting. 

## **5. Dataset Splitting and Class Balancing**

To prepare the dataset for training:

**Splitting:**
-**70% Training**
-**15% Validation**
-**15% Testing**
Validation and test sets remained **unbalanced** to reflect real-world conditions.

## **6. Training Oversampling:**

Class imbalance (especially for organic and functional conditions) was addressed using **Random Oversampling only on the training set.**
This ensured:
-Equal number of samples for each class
-Fair learning and improved classifier stability
Final counts:
-**Binary classification:** 4,353 images per class
-**Tertiary classification:** 4,633 images per class

This resulted in a robust and well-structured training dataset.
## **7. Model Training and Architecture Comparison**
- Multiple deep learning architectures were implemented:
  - **InceptionV3**
  - **DenseNet121**
  - **ResNet50V2**
  - **AlexNet**
  - **ConvNeXt V2**
- All models were trained on **preprocessed, segmented, augmented, and balanced kymogram datasets**.
- Each architecture was configured specifically for:
  - **Binary classification:** Healthy vs Unhealthy
  - **Tertiary classification:** Healthy, Functional, Organic
- All models were trained for **30 epochs** with optimized hyperparameters and regularization techniques.

  
## DenseNet121 – Training Pipeline
- Input images resized to **224×224 px** for uniformity.
- Batch size **32** to optimize GPU utilization.
- DenseNet121 pretrained on **ImageNet**, used as a **frozen feature extractor**.
- Output of backbone → **Flatten layer** → 1D vector.
- Fully connected classifier:
  - Dense (512 units, ReLU)
  - Dense (256 units, ReLU)
  - Batch Normalization
  - Dropout
  - Final Dense → **1 or 3 output units** (binary/tertiary)
- Final activation: **Sigmoid** (binary) or **Softmax** (tertiary).
- Achieves efficient feature extraction + strong generalization.

---

## AlexNet (with ResNet-152 Feature Extractor)
- ResNet-152 used as **fixed feature extractor** (last layer removed).
- Extracts **2048-dimensional** deep features.
- Custom AlexNet-style classifier:
  - Two Dense layers → **4096 units** each (ReLU)
  - Dropout **0.5** for regularization
  - Output layer with **2 or 3 units**
- Trained for **30 epochs** using **Adam optimizer** + cross-entropy loss.
- Demonstrated high accuracy and strong generalization.

---

## EfficientNetV2
- Inputs resized to **224×224** with horizontal flip augmentation.
- Applied **Exponential Moving Average (EMA)** to smooth model weights.
- Architecture includes:
  - Stem convolution (3×3, 32 filters)
  - 5 MBConv blocks with progressive scaling:
    - Filters: 32 → 64 → 128 → 256 → 512
  - GELU activation throughout
  - Final 1280-filter Conv layer → BatchNorm → GELU
  - Classification head → **Softmax**
- Trained for 30 epochs with:
  - Adam optimizer (lr=1e-4)
  - Cross-entropy loss
  - Best-weights checkpointing
- ~22 million parameters, highly efficient and scalable.

---

## ResNet50V2 (with ResNet-152 Features + MLP)
- ResNet-152 backbone extracts **2048-D deep features**.
- Custom residual classifier includes:
  - Two FC layers per block
  - BatchNorm + ReLU
  - Skip connections (ResNet-style)
  - Dropout (0.3)
- Final Dense layer outputs **2 or 3 units**.
- Trained for 30 epochs using Adam (lr=1e-4) + cross-entropy loss.
- Residual MLP improves gradient flow & reduces overfitting.

---

## ConvNeXt V2 Hybrid Model
- Backbone: **ResNet-152 (frozen)**.
- Classifier: ConvNeXt V2-inspired MLP head:
  - LayerNorm
  - Dense (1024 units, GELU)
  - Dense (512 units, GELU)
  - Dropout (0.3)
- Optimizer: **AdamW** (lr=1e-4, weight decay=1e-4).
- Learning rate scheduler: **ReduceLROnPlateau**.
- Effective for both binary & tertiary tasks.

---

## InceptionV3 (Best Model)
- Input resized to **299×299**.
- Applied data augmentation:
  - Color jitter
  - Random horizontal flip
  - Small rotations
- Inception modules extract multi-scale features:
  - Parallel 1×1, 3×3, 5×5 convolutions
- Includes **auxiliary classifiers** to stabilize deep gradients.
- Feature maps passed through **Global Average Pooling (GAP)** → Linear → Output.
- Final layer adapted for **2-class or 3-class** prediction.
- Trained using **Adam optimizer (1e-4)** + weighted cross-entropy.
- Achieved **best accuracy in both binary and tertiary classification**.

---

## Summary of Training Insights
- All models were trained independently for binary and tertiary classification.
- InceptionV3 and ResNet50V2 consistently produced **highest accuracy + best AUC scores**.
- DenseNet121, EfficientNetV2, AlexNet, and ConvNeXtV2 also showed high performance.
- Custom 2D-CNN (baseline) achieved strong tertiary results without transfer learning.

