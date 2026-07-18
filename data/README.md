# FPGA-Accelerated AI-Based Stroke Rehabilitation System Using EEG Motor Imagery Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](requirements.txt)

## Repository Overview

This repository contains the development of an FPGA-accelerated Edge AI framework for EEG-based motor imagery recognition aimed at assisting post-stroke rehabilitation. The project evaluates both traditional machine learning and lightweight deep learning approaches using publicly available acute stroke EEG datasets, with the long-term goal of enabling real-time deployment on FPGA-based wearable rehabilitation devices.

For a deeper dive into the methodology, algorithms, and proposed hardware architectures, please read the [Technical Documentation](TECHNICAL_DOCUMENTATION.md). For details on the dataset structure and acquisition, please refer to the [Dataset Guide](DATASET_GUIDE.md).

---

## Novelty

*   **Clinical Dataset Focus**: Evaluation of traditional EEG signal processing and modern deep learning models specifically on acute stroke EEG data, which exhibits highly unique, non-stationary electrical distortions compared to typical healthy-subject datasets.
*   **Edge-AI Feasibility**: Investigation of lightweight architectures (such as EEGNet and compact 1D-CNNs) that are computationally optimized for edge-hardware deployment.
*   **Hardware-Software Co-Design**: A proposed pipeline designed from the ground up for low-latency hardware acceleration (targeting FPGA fabrics) to achieve sub-10ms latency.
*   **Rehabilitation Paradigm**: Establishing a translation pathway from neural motor imagery classification (Left vs. Right hand grasping) to active control of wearable rehabilitation devices (e.g., robotic exoskeleton gloves).

---

## Research Roadmap

The development lifecycle of this framework is structured as follows:

```text
Current Stage: Active Software Validation & Modeling
[Dataset Analysis] ✔ ──> [EEG Preprocessing] ✔ ──> [Traditional ML Models] ──> [Deep Learning Models]
                                                                                      │
[Wearable Integration] <── [Hardware Validation] <── [FPGA Deployment] <── [Model Comparison]
```

*   **Phase 1: Dataset Sourcing & Analysis** ✔: Sourced and structured the clinical acute stroke dataset.
*   **Phase 2: Signal Preprocessing Engine** ✔: Built modular bandpass filtering, notch filtering, and Independent Component Analysis (ICA) components for eye-blink artifact subtraction.
*   **Phase 3: Traditional Machine Learning Benchmarks**: Evaluating Common Spatial Patterns (CSP) + LDA/SVM, Filter Bank CSP (FBCSP), and Riemannian Geometry models (MDRM, Tangent Space).
*   **Phase 4: Deep Learning Optimization**: Implementing and training lightweight temporal-spatial neural networks (EEGNet, 1D-CNN).
*   **Phase 5: FPGA Acceleration & Deployment**: Mapping the validated algorithms to hardware architectures (custom RTL, DSP filters, and DPUs).
*   **Phase 6: Closed-Loop Wearable Integration**: Linking the FPGA control signals to robotic hand actuators for physical patient feedback.

---

## Current Status & Limitations

The FPGA hardware implementation and the mechanical wearable rehabilitation glove are proposed future works. The current codebase focuses on signal processing, feature extraction, neural network training, and software validation. Hardware deployment structures outlined in the [Technical Documentation](TECHNICAL_DOCUMENTATION.md) serve as the design specification for the next phase.

---

## Proposed Repository Structure

To support clean codebase management, files are organized into logical partitions:

| Directory/File | Description |
| :--- | :--- |
| **[`TECHNICAL_DOCUMENTATION.md`](TECHNICAL_DOCUMENTATION.md)** | Technical pipeline details, model definitions, and the proposed FPGA hardware design. |
| **[`DATASET_GUIDE.md`](DATASET_GUIDE.md)** | Clinical cohort statistics, channel layout, and data parsing details. |
| **[`requirements.txt`](requirements.txt)** | Python environment dependency package list. |
| **[`LICENSE`](LICENSE)** | MIT open-source license. |
| **`/notebooks/`** | Contains Jupyter Notebooks (e.g., [`EEG_BCI_MI_Pipeline.ipynb`](New%20folder/EEG_BCI_MI_Pipeline.ipynb)) for interactive development. |
| **`/src/`** or **`/code/`** | Core scripts containing the pipeline methods ([`Final_Colab_Master.py`](New%20folder/Final_Colab_Master.py)). |
| **`/docs/`** | Supplementary reading, block diagrams, and explainers. |
| **`/fpga/`** | *(Proposed)* Hardware description files (Verilog/VHDL) and PYNQ deployment notebooks. |

---

## Getting Started

1. Clone the repository and navigate into the root directory.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the interactive pipeline notebook in your local environment or Google Colab:
   ```bash
   jupyter notebook notebooks/EEG_BCI_MI_Pipeline.ipynb
   ```

---

## Future Work

*   **Quantization-Aware Training (QAT)**: Compressing EEGNet parameters down to 8-bit integers (INT8) to fit on low-power edge FPGAs with negligible accuracy loss.
*   **Transfer Learning**: Developing subject-independent classifiers using domain adaptation to eliminate the need for lengthy patient-specific calibration phases.
*   **FPGA Inference Core**: Implementing a custom Verilog accelerator for the Conv2D and DepthwiseConv2D layers of EEGNet, or deploying using AMD Xilinx DPU architectures.
*   **Closed-Loop Rehabilitation**: Building a mechanical glove controlled directly by real-time motor imagery triggers (e.g., imagining grasp opens/closes the hand).

---

## Citation

If this repository contributes to your work, please cite the underlying dataset and study:

```text
Haijie Liu et al.
An EEG motor imagery dataset for brain-computer interface in acute stroke patients.
Scientific Data, Nature, 2024.
https://doi.org/10.1038/s41597-023-02787-8
```
