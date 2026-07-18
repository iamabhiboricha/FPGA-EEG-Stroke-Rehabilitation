# Dataset Guide: Clinical Acute Stroke EEG Motor Imagery Dataset

This document details the structure, demographics, channel mapping, and data loading instructions for the clinical acute stroke motor imagery dataset utilized in this project.

---

## 1. Sourcing & Reference

*   **Dataset Name**: *An EEG motor imagery dataset for brain-computer interface in acute stroke patients*
*   **Authors**: Haijie Liu, Penghu Wei, Haochong Wang, et al.
*   **Journal**: *Scientific Data*, Nature, 2024
*   **Doi Link**: [https://doi.org/10.1038/s41597-023-02787-8](https://doi.org/10.1038/s41597-023-02787-8)
*   **Significance**: This is the first publicly available, open-access dataset addressing left- and right-handed motor imagery (MI) specifically in acute stroke patients, representing a major resource for clinical BCI applications.

---

## 2. Clinical Cohort Demographics

The dataset consists of recordings from **50 acute stroke patients** (hospitalized at Xuanwu Hospital of Capital Medical University, Beijing, China).

| Characteristic | Cohort Metric (N=50) |
| :--- | :--- |
| **Sex** | 39 Male (78%) / 11 Female (22%) |
| **Age** | 31 to 77 years (Mean: 56.70, SD: 10.57) |
| **Stroke Onset** | 1 to 30 days post-onset (Mean: 5.78 days) |
| **Hemiplegic Side** | 27 Left Paralysis (54%) / 23 Right Paralysis (46%) |
| **NIHSS Score** | Mean: 4.16, SD: 2.85 *(National Institute of Health Stroke Scale)* |
| **Modified Barthel Index (MBI)** | Mean: 70.94, SD: 18.22 *(Functional independence)* |
| **Modified Rankin Scale (mRS)** | Mean: 2.66, SD: 1.44 *(Overall disability)* |

---

## 3. Data Collection Hardware & Channels

*   **Acquisition System**: Wireless multichannel EEG acquisition cap (ZhenTec NT1).
*   **Sampling Rate**: 500 Hz.
*   **EEG Layout**: 29 EEG channels placed according to the international 10-10 system:
    *   *Frontal*: Fp1, Fp2, Fz, F3, F4, F7, F8, FCz, FC3, FC4, FT7, FT8
    *   *Central/Temporal*: Cz, C3, C4, T7, T8, CPz, CP3, CP4, TP7, TP8
    *   *Parietal/Occipital*: Pz, P3, P4, P7, P8, Oz, O1, O2
*   **Ocular Reference Channels (EOG)**: 
    *   `HEOL` (Horizontal Electrooculogram Left)
    *   `HEOR` (Horizontal Electrooculogram Right)
    *   *Used to track eye-blink artifacts for Independent Component Analysis (ICA) subtraction.*
*   **Stimulus Channel**:
    *   `STI` (Stimulator channel)
    *   *Records voltage spikes indicating trial events (instruction, motor imagery start, break).*

---

## 4. Repository Directory Structure

To run the pipeline, the downloaded dataset files should be organized as follows relative to your workspace:

```text
/data
├── /edffile/
│   ├── /sub-01/
│   │   └── /eeg/
│   │       └── sub-01_task-motor-imagery_eeg.edf   <-- Raw continuous brainwaves
│   └── /sub-50/
│       └── /eeg/
│           └── sub-50_task-motor-imagery_eeg.edf
└── /sourcedata/
    └── /sourcedata/
        ├── /sub-01/
        │   └── sub-01_task-motor-imagery_eeg.mat   <-- Ground truth left/right labels
        └── /sub-50/
            └── sub-50_task-motor-imagery_eeg.mat
```

*   **EDF files (`.edf`)**: Stores continuous multichannel recordings.
*   **MAT files (`.mat`)**: Stores an array containing sequential class labels:
    *   `1` represents imagining **Left-Hand** gripping.
    *   `2` represents imagining **Right-Hand** gripping.

---

## 5. Reading and Mapping the Dataset

The continuous brainwave signals (EDF) and the labels (MAT) are synchronized by detecting voltage spikes on the `STI` channel. Each run consists of **40 trials** (20 left-hand, 20 right-hand tasks).

Below is a Python snippet using `MNE` and `SciPy` to load and segment the trials:

```python
import mne
import scipy.io as sio
import numpy as np

def load_patient_epochs(sub_id, edf_path, mat_path):
    # 1. Load raw EDF file
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw.rename_channels({'': 'STI'}) # Rename empty-labeled trigger channel to STI
    raw.set_channel_types({'HEOL': 'eog', 'HEOR': 'eog', 'STI': 'stim'})
    
    # 2. Extract Ground Truth Labels from MAT file
    # Subtract 1 to convert labels 1 and 2 to index 0 (Left) and 1 (Right)
    labels = sio.loadmat(mat_path)['eeg']['label'][0, 0].flatten() - 1 
    
    # 3. Detect Stimulus Triggers in the STI Channel
    # Trigger Marker 2 (imagining stage) shows a spike between 1.5uV and 2.5uV
    stim_channel_data = raw.get_data(picks=['STI'])[0]
    spike_indices = np.where((stim_channel_data > 1.5e-6) & (stim_channel_data < 2.5e-6))[0]
    
    # Filter overlapping consecutive samples of the same spike (minimum 1000 sample gap)
    starts = np.insert(np.diff(spike_indices) > 1000, 0, True)
    valid_spike_indices = spike_indices[starts][:len(labels)]
    
    # 4. Construct MNE Event Matrix
    # Event format: [sample_index, 0, event_id]
    events = np.column_stack((
        valid_spike_indices, 
        np.zeros(len(valid_spike_indices), dtype=int), 
        np.ones(len(valid_spike_indices), dtype=int)
    ))
    
    # 5. Segment into 4-second trials (Epochs)
    # Motor imagery happens between t=0s and t=4s following the cue
    epochs = mne.Epochs(raw, events, event_id=1, tmin=0, tmax=4, baseline=None, preload=True)
    epochs_eeg = epochs.copy().pick_types(eeg=True, eog=False, stim=False)
    
    return epochs_eeg.get_data(), labels, epochs_eeg.ch_names
```
