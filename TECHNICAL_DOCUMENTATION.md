# Technical Documentation: EEG Motor Imagery Processing & Modeling Pipeline

This document provides a detailed technical overview of the signal processing, feature engineering, classification models, and the proposed FPGA hardware-software co-design for the **FPGA-Accelerated AI-Based Stroke Rehabilitation System**.

---

## 1. End-to-End Signal Processing Pipeline

The system is designed around a rigorous processing pipeline that converts continuous, noisy scalp voltage measurements into robust control signals.

```text
       +-----------------------------+
       | Raw EEG Signal (.edf)       |
       +--------------+--------------+
                      |
                      v
       +--------------+--------------+
       | 8-30 Hz Bandpass Filter     | (Alpha/Beta isolation)
       +--------------+--------------+
                      |
                      v
       +--------------+--------------+
       | ICA Blink Artifact Removal  | (Correlation with HEOL/HEOR)
       +--------------+--------------+
                      |
                      v
       +--------------+--------------+
       | Event Extraction (STI)      | (Thresholding 1.5uV - 2.5uV)
       +--------------+--------------+
                      |
                      v
       +--------------+--------------+
       | Epoching (0 - 4s trials)    | (30 channels extracted)
       +-----------------------------+
```

### A. Spectral Filtering
*   **Target Frequency Bands**: Motor Imagery (imagining movement) triggers event-related desynchronization (ERD) and event-related synchronization (ERS) in the **Alpha (Mu: 8–13 Hz)** and **Beta (14–30 Hz)** frequency bands over the motor cortex.
*   **Filter Type**: A zero-phase 4th-order Butterworth bandpass filter (8–30 Hz) is applied to continuous data. This isolates the physiological frequencies of interest and removes slow DC drifts (sweating, slow movements) and high-frequency EMG (muscle) noise.
*   **Notch Filtering (50 Hz)**: Analysis of the source EDF files shows the signals were pre-filtered to 0.5–40 Hz. Because 50/60 Hz powerline interference is already physically absent, a notch filter is mathematically redundant.

### B. Ocular Artifact Rejection (ICA)
*   **The Problem**: Eye blinks create massive voltage spikes (EOG artifacts) that propagate across the frontal scalp channels, blinding classifiers.
*   **The Solution**: Independent Component Analysis (ICA) separates the mixed multi-channel EEG signals into 15 mathematically independent sources.
*   **Automation**: The pipeline computes Pearson correlation coefficients between each independent component and the physical EOG channels (`HEOL` and `HEOR`). Components exceeding the bad threshold are marked as ocular noise and mathematically subtracted. The clean EEG is reconstructed using only brain-related components.

---

## 2. Epoch Segmentation & Safe Augmentation

A primary challenge with stroke datasets is the limited number of clinical trials per patient (typically 40). Deep learning models require larger datasets, making data augmentation necessary.

### A. Stimulus Trigger Mapping
The raw continuous file contains a stimulus channel (`STI`). The pipeline scans for spikes in the range of `1.5e-6 V` to `2.5e-6 V`. This voltage threshold locks onto the precise onset of the Motor Imagery stage (Marker 2), ignoring calibration and break markers. Once the onset sample index is found, the signal is cut into 4-second blocks (Epochs).

### B. Preventing Data Leakage
Slicing epochs into overlapping windows increases the sample size but poses a high risk of **data leakage** if done in the wrong sequence:

```text
INCORRECT (Leakage!):
[All Epochs] ──> [Slice into Overlapping Windows] ──> [Random Train/Test Split]
(Overlapping windows from the same trial end up in both Train and Test, inflating accuracy artificially)

CORRECT (Strict Isolation):
[All Epochs] ──> [Stratified K-Fold Split] ──> [Slice Train Epochs] ──> [Train Model]
                                           ──> [Slice Test Epochs]  ──> [Evaluate Model]
```

*   **Implementation**: Stratified 5-Fold Cross-Validation is applied directly to the *whole 4-second trials*.
*   **Augmentation**: For each fold, 2-second sliding windows with a 1-second step are extracted from the trials. This increases the training size threefold (from 32 to 96 samples) and testing size threefold (from 8 to 24 samples), keeping the testing set strictly isolated.
*   **Normalization**: Z-score normalization (mean = 0, standard deviation = 1) is applied over the time dimension of each slice to ensure stable neural network training.

---

## 3. Evaluated Classification Frameworks

The repository is built to benchmark traditional machine learning against compact deep learning.

### A. Traditional Machine Learning Pipeline

#### 1. Common Spatial Patterns (CSP) + LDA
*   **Algorithm**: CSP finds spatial filters that maximize signal variance for Class 1 (Left Hand) while minimizing variance for Class 2 (Right Hand).
*   **Dimensionality Constraint**: Traditional CSP is highly sensitive to noise. When processing 30 channels with only ~30 training trials, the covariance matrix becomes ill-conditioned (the "Curse of Dimensionality"). To prevent this, the input is restricted to **7 motor cortex channels** (`C3`, `C4`, `Cz`, `CP3`, `CP4`, `FC3`, `FC4`).
*   **Classifier**: The log-variance features of the filtered signals are classified using Linear Discriminant Analysis (LDA).

#### 2. Filter Bank CSP (FBCSP) + SVM
*   The 7-channel signal is decomposed into 5 sub-bands (8–12, 12–16, 16–20, 20–24, 24–30 Hz).
*   CSP is computed for each band. Feature selection is performed using **Mutual Information (SelectKBest)** to identify the 10 most discriminative features across all bands, which are then classified using a Support Vector Machine (SVM) with an RBF kernel.

#### 3. Riemannian Geometry (MDM & Tangent Space SVM)
*   **Concept**: EEG covariance matrices lie on a curved, high-dimensional space (the Riemannian manifold). Riemannian methods map these matrices directly to preserve their physical and geometric properties.
*   **MDM**: Calculates the Riemannian mean covariance matrix for each class and classifies new trials using the Minimum Distance to the Mean.
*   **Tangent Space SVM (FBTS)**: Projects covariance matrices onto a flat tangent space where they can be classified using a linear SVM.

---

### B. Deep Learning Architectures

Deep learning models automate spatial and temporal feature extraction. They are fed the **full 30-channel montage** because spatial convolutions and dropout layers naturally regularize the network and prevent overfitting.

#### 1. EEGNet
A compact convolutional neural network tailored for BCI applications:
*   **Temporal Convolution**: A 2D convolutional filter (`1 x Samples/2`) that acts as a bandpass filter over time.
*   **Depthwise Spatial Convolution**: A depthwise filter (`Chans x 1`) that acts as a spatial filter, learning electrode weights similar to CSP.
*   **Separable Convolution**: Combines spatial and temporal representations while minimizing the total parameter count.

#### 2. 1D-CNN
A model that treats the 30 EEG channels as parallel time-series features. It uses parallel 1D convolutional layers, batch normalization, max pooling, and dropout to extract temporal patterns.

#### 3. Overfitting Prevention
EEG data from acute stroke patients is highly non-stationary. Both networks use an `EarlyStopping` callback that monitors `val_accuracy` (patience = 5 or 15) and restores the best weights, saving the model at its peak validation accuracy before it begins to overfit to background noise.

---

## 4. Proposed FPGA Acceleration Architecture

For real-time assistive stroke rehabilitation, the inference pipeline must run with ultra-low latency on portable, low-power edge hardware. Below is the proposed hardware architecture for future FPGA implementation:

```text
              RAW EEG DIGITIZED CHANNELS (SPI/I2S Interface)
                                  │
                                  v
                    +───────────────────────────+
                    │  Multi-Channel FIR Filter │ (8-30 Hz bandpass)
                    │  (Parallel DSP Slices)    │
                    +─────────────┬─────────────+
                                  │
                                  v
                    +───────────────────────────+
                    │    CSP Projection Core    │ (Systolic Array Matrix Multiplier)
                    │     Y = W^T * X           │
                    +─────────────┬─────────────+
                                  │
                                  v
                    +───────────────────────────+
                    │ Feature Extraction Engine │ (Variance computation & log-approximation)
                    │  Var(y) = Sum(y^2) / N    │
                    +─────────────┬─────────────+
                                  │
                                  v
                    +───────────────────────────+
                    │ Classification Processor  │ (Parallel MAC units for LDA/SVM)
                    │   Score = W * F + B       │
                    +─────────────┬─────────────+
                                  │
                                  v
                    REHABILITATION TRIGGER SIGNAL (GPIO)
```

### A. Multi-Channel Bandpass Filtering Core
*   **Implementation**: A bank of parallel Finite Impulse Response (FIR) filters.
*   **Hardware Mapping**: Leverages the FPGA’s internal **DSP Slices** (MAC units) to perform concurrent multiplications of incoming EEG channels with hardcoded filter coefficients.

### B. CSP Spatial Filter Systolic Array
*   **Implementation**: The spatial filtering operation $Y = W^T X$ is a matrix multiplication where $W$ is the pre-trained CSP spatial filter matrix, and $X$ is the filtered EEG signal matrix.
*   **Hardware Mapping**: A 2D Systolic Array of processing elements (PEs) that shifts data through registers, maximizing throughput and reducing memory access bottlenecks.

### C. Feature Extraction & Logarithm Unit
*   **Variance Calculation**: Computes the sum of squares of each output channel over the window length. This is implemented using hardware squarers and accumulators.
*   **Log Approximation**: To avoid expensive floating-point division and natural logarithm calculations on-chip, a lookup table (LUT) or CORDIC (Coordinate Rotation Digital Computer) algorithm is used to compute the log-variance features.

### D. Classification Processor (LDA/SVM)
*   **LDA/Linear SVM**: The final decision function $y = \mathbf{w}^T \mathbf{f} + b$ is implemented using parallel Multiply-Accumulate (MAC) units. The result is thresholded to output a single-bit control line.
*   **EEGNet Acceleration**: For the deep learning pathway, the model can be compiled onto an **AMD Xilinx Deep Learning Processor Unit (DPU)** or custom CNN accelerator IP cores using high-level synthesis (HLS).

### E. Actuator Driver Interface
*   The single-bit classification output is mapped to a physical GPIO pin on the FPGA. This pin drives a relay or servo controller to actuate a pneumatic robotic glove, opening or closing the patient's hand based on their detected motor imagery.
