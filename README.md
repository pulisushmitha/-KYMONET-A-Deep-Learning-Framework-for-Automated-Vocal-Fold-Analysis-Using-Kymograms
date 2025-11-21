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


**2. Kymogram Generation Using MATLAB**
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

3. Video Enhancement using RIFE and FFmpeg

Some HSV videos (29 samples) originally produced **noisy or blurred kymograms** due to issues such as motion blur, excessive brightness, or low frame uniformity.

To correct this, two enhancement methods were used:

**3.1 RIFE (Real-Time Intermediate Flow Estimation)**

RIFE is a deep learning–based frame interpolation model that generates intermediate frames between existing ones.
Benefits:
-Smoothens transitions
-Reduces motion artifacts
-Produces clearer vibratory sequences

After RIFE enhancement, videos were more stable and resulted in improved kymograms. However, only **5 out of 29 videos** were successfully stabilized; the remaining suffered from extreme noise and were discarded based on clinical research practices. 

**3.2 FFmpeg Slow-Motion Processing**

Videos were slowed down to **0.2× speed** using FFmpeg to increase temporal resolution and allow more accurate scanline extraction. This preprocessing helped reduce inconsistencies caused by rapid oscillations of vocal folds. 

**4. Kymogram Segmentation and Augmentation**
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

**5. Dataset Splitting and Class Balancing**

To prepare the dataset for training:

**Splitting:**
-**70% Training**
-**15% Validation**
-**15% Testing**
Validation and test sets remained **unbalanced** to reflect real-world conditions.

**Training Oversampling:**

Class imbalance (especially for organic and functional conditions) was addressed using **Random Oversampling only on the training set.**
This ensured:
-Equal number of samples for each class
-Fair learning and improved classifier stability
Final counts:
-**Binary classification:** 4,353 images per class
-**Tertiary classification:** 4,633 images per class

This resulted in a robust and well-structured training dataset.
