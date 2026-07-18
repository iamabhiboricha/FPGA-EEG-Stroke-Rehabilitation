# ==============================================================================
# COPY AND PASTE EACH OF THESE BLOCKS INTO A SEPARATE CELL IN GOOGLE COLAB
# ==============================================================================

# %% [markdown]
# # Cell 1: Environment Setup & Library Imports
# *Run this cell first to install required libraries and import dependencies.*

# %%
!pip install pyriemann mne scikit-learn

import os
import mne
import numpy as np
import scipy.io as sio
import pandas as pd
import tensorflow as tf

from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from mne.decoding import CSP

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("✅ GPU successfully initialized!")


# %% [markdown]
# # Cell 2: Data Exploration & Visualization (EDA)
# *Run this cell to generate beautiful plots that prove to your mentor you actively explored the signal data.*

# %%
sub_id = "01"
edf_path = f"/content/drive/MyDrive/edffile/sub-{sub_id}/eeg/sub-{sub_id}_task-motor-imagery_eeg.edf"

if os.path.exists(edf_path):
    import matplotlib.pyplot as plt
    print(f"Loading Subject {sub_id} for Data Exploration...")
    raw_eda = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw_eda.rename_channels({'': 'STI'})
    raw_eda.set_channel_types({'HEOL': 'eog', 'HEOR': 'eog', 'STI': 'stim'})
    
    # 1. Print the scientific data structure
    print("\n=============================================")
    print("             DATASET INFORMATION             ")
    print("=============================================")
    print(raw_eda.info)
    print("\nChannel Names:", raw_eda.ch_names)
    
    # 2. Set the standard 10-20 Brain Montage and Plot Sensor Locations
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw_eda.set_montage(montage, on_missing='ignore')
        fig = raw_eda.plot_sensors(show_names=True, title="EEG Sensor Map (10-20 System)")
        plt.show()
    except Exception as e:
        print("Could not plot montage.")
        
    # 3. Plot the raw continuous brainwaves (first 10 seconds)
    fig_raw = raw_eda.plot(duration=10, n_channels=15, title="Raw EEG Brainwaves (First 10s)", show=False)
    plt.show()
    
    # 4. Plot the Power Spectral Density (PSD) to show the frequency distribution
    print("\nCalculating Frequency Power Spectrum...")
    fig_psd = raw_eda.compute_psd(fmax=40, verbose=False).plot(show=False)
    plt.show()
else:
    print("Data path not found. Please ensure Google Drive is mounted!")

# %% [markdown]
# # Cell 3: Modular Architecture & Preprocessing Engine
# *This cell contains the robust helper functions. Organizing code into functions is an industry standard that proves you know how to write DRY (Don't Repeat Yourself) code.*

# %%
def create_overlapping_windows(X, y, window_size=1000, step_size=500):
    """Data Augmentation via Time Shifting"""
    X_out, y_out = [], []
    for i in range(len(X)):
        for start in range(0, X.shape[2] - window_size + 1, step_size):
            X_out.append(X[i, :, start:start+window_size])
            y_out.append(y[i])
    return np.array(X_out), np.array(y_out)

def normalize_data(X):
    """Z-score normalization strictly for Deep Learning"""
    mean = np.mean(X, axis=2, keepdims=True)
    std = np.std(X, axis=2, keepdims=True)
    std[std == 0] = 1e-8
    return (X - mean) / std

def apply_filter_bank(X, fs=500):
    """Slices EEG into 5 frequency bands for Advanced FB Mathematics"""
    bands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    X_fb = []
    for low, high in bands:
        b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='bandpass')
        X_fb.append(filtfilt(b, a, X, axis=2))
    return X_fb

def extract_fbcsp_features(X_fb_train, X_fb_test, y_train):
    """Mutual Information Feature Selection for FBCSP"""
    X_tr, X_te = [], []
    for X_b_train, X_b_test in zip(X_fb_train, X_fb_test):
        csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
        X_tr.append(csp.fit_transform(X_b_train, y_train))
        X_te.append(csp.transform(X_b_test))
    
    selector = SelectKBest(mutual_info_classif, k=10)
    return selector.fit_transform(np.hstack(X_tr), y_train), selector.transform(np.hstack(X_te))

def build_eegnet(nb_classes=1, Chans=30, Samples=1000, dropoutRate=0.5):
    input1 = Input(shape=(Chans, Samples, 1))
    b1 = Conv2D(8, (1, 250), padding='same', use_bias=False)(input1)
    b1 = BatchNormalization(axis=-1)(b1)
    b1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.))(b1)
    b1 = BatchNormalization(axis=-1)(b1)
    b1 = Activation('elu')(b1)
    b1 = AveragePooling2D((1, 4))(b1)
    b1 = Dropout(dropoutRate)(b1)
    b2 = SeparableConv2D(16, (1, 16), use_bias=False, padding='same')(b1)
    b2 = BatchNormalization(axis=-1)(b2)
    b2 = Activation('elu')(b2)
    b2 = AveragePooling2D((1, 8))(b2)
    b2 = Dropout(dropoutRate)(b2)
    out = Activation('sigmoid')(Dense(nb_classes, kernel_constraint=max_norm(0.25))(Flatten()(b2)))
    model = Model(inputs=input1, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_1d_cnn(nb_classes=1, Chans=30, Samples=1000):
    model = Sequential([
        Input(shape=(Samples, Chans)),
        Conv1D(64, 20, activation='relu'), BatchNormalization(), MaxPooling1D(4), Dropout(0.5),
        Conv1D(128, 10, activation='relu'), BatchNormalization(), MaxPooling1D(4), Dropout(0.5),
        Flatten(), Dense(64, activation='relu'), Dropout(0.5),
        Dense(nb_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_subject_data(sub_id):
    edf_path = f"/content/drive/MyDrive/edffile/sub-{sub_id}/eeg/sub-{sub_id}_task-motor-imagery_eeg.edf"
    mat_path = f"/content/drive/MyDrive/sourcedata/sourcedata/sub-{sub_id}/sub-{sub_id}_task-motor-imagery_eeg.mat"
    if not os.path.exists(edf_path): return None, None, None
        
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels({'': 'STI'})
    raw.set_channel_types({'HEOL': 'eog', 'HEOR': 'eog', 'STI': 'stim'})
    
    # 40Hz Pre-filtered in dataset, 50Hz powerline is completely absent
    raw.filter(l_freq=8., h_freq=30., picks=['eeg'], verbose=False) # Isolate MI bands
    
    ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=42)
    ica.fit(raw, picks=['eeg'], verbose=False)
    eog_indices, _ = ica.find_bads_eog(raw, ch_name=['HEOL', 'HEOR'], verbose=False)
    ica.exclude = eog_indices
    ica.apply(raw, verbose=False)
    
    labels = sio.loadmat(mat_path)['eeg']['label'][0, 0].flatten() - 1 
    
    stim = raw.get_data(picks=['STI'])[0]
    pos = np.where((stim > 1.5e-6) & (stim < 2.5e-6))[0]
    if len(pos) == 0: return None, None, None
    valid_pos = pos[np.insert(np.diff(pos) > 1000, 0, True)][:min(len(pos), len(labels))]
    
    events = np.column_stack((valid_pos, np.zeros(len(valid_pos), dtype=int), np.ones(len(valid_pos), dtype=int)))
    epochs = mne.Epochs(raw, events, event_id=1, tmin=0, tmax=4, baseline=None, preload=True, verbose=False)
    epochs_eeg = epochs.copy().pick_types(eeg=True, eog=False, stim=False)
    
    return epochs_eeg.get_data(), labels[:len(valid_pos)], epochs_eeg.ch_names


# %% [markdown]
# # Cell 4: The Grand Evaluation Loop
# *Evaluates all 6 models simultaneously per subject to optimize RAM usage and prevent Google Colab timeouts.*

# %%
subjects = [f"{i:02d}" for i in range(1, 51)]
csv_path = "/content/drive/MyDrive/Final_BCI_Benchmarks.csv"

# Safe Resume Logic
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    completed = df['Subject'].astype(str).str.zfill(2).tolist()
    scores = df.drop('Subject', axis=1).to_dict('list')
    print(f"Resuming! Already finished {len(completed)} subjects.")
else:
    completed = []
    scores = {'CSP_LDA':[], 'FBCSP':[], 'MDM':[], 'FBTS':[], 'EEGNet':[], 'CNN_1D':[]}

for sub in subjects:
    if sub in completed: continue
    X, y, ch_names = load_subject_data(sub)
    if X is None: continue
        
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s_scores = {k: [] for k in scores.keys()}
    
    for tr, te in kf.split(X, y):
        # Base Split
        X_tr_raw, X_te_raw, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        
        # Augmentation (Applies to all)
        X_tr_aug, y_tr_aug = create_overlapping_windows(X_tr_raw, y_tr)
        X_te_aug, y_te_aug = create_overlapping_windows(X_te_raw, y_te)
        
        # =======================================================
        # 1. TRADITIONAL PIPELINE (No Normalization, 7 Channels)
        # =======================================================
        m_idx = [ch_names.index(c) for c in ['C3','C4','Cz','CP3','CP4','FC3','FC4'] if c in ch_names]
        X_tr_m, X_te_m = X_tr_aug[:, m_idx, :], X_te_aug[:, m_idx, :]
        
        # CSP + LDA
        clf = Pipeline([('CSP', CSP(4, reg='ledoit_wolf', log=True)), ('LDA', LinearDiscriminantAnalysis())])
        clf.fit(X_tr_m, y_tr_aug)
        s_scores['CSP_LDA'].append(accuracy_score(y_te_aug, clf.predict(X_te_m)))
        
        # FBCSP
        X_fb_tr, X_fb_te = apply_filter_bank(X_tr_m), apply_filter_bank(X_te_m)
        X_tr_fbcsp, X_te_fbcsp = extract_fbcsp_features(X_fb_tr, X_fb_te, y_tr_aug)
        svm = SVC(kernel='rbf', C=1.0).fit(X_tr_fbcsp, y_tr_aug)
        s_scores['FBCSP'].append(accuracy_score(y_te_aug, svm.predict(X_te_fbcsp)))
        
        # Riemannian MDM
        mdm = Pipeline([('Cov', Covariances('oas')), ('MDM', MDM(metric=dict(mean='riemann', distance='riemann')))])
        mdm.fit(X_tr_m, y_tr_aug)
        s_scores['MDM'].append(accuracy_score(y_te_aug, mdm.predict(X_te_m)))
        
        # Filter Bank Tangent Space (FBTS)
        X_tr_ts, X_te_ts = [], []
        for b_tr, b_te in zip(X_fb_tr, X_fb_te):
            ts = TangentSpace(metric='riemann')
            cov = Covariances('oas')
            X_tr_ts.append(ts.fit_transform(cov.fit_transform(b_tr)))
            X_te_ts.append(ts.transform(cov.transform(b_te)))
        svm_ts = SVC(kernel='linear', C=1.0).fit(np.hstack(X_tr_ts), y_tr_aug)
        s_scores['FBTS'].append(accuracy_score(y_te_aug, svm_ts.predict(np.hstack(X_te_ts))))
        
        # =======================================================
        # 2. DEEP LEARNING PIPELINE (Normalization, 30 Channels)
        # =======================================================
        X_tr_dl, X_te_dl = normalize_data(X_tr_aug), normalize_data(X_te_aug)
        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)
        
        # EEGNet
        tf.keras.backend.clear_session()
        eegnet = build_eegnet(Chans=30, Samples=X_tr_dl.shape[2])
        eegnet.fit(X_tr_dl[..., np.newaxis], y_tr_aug, epochs=50, batch_size=32, validation_data=(X_te_dl[..., np.newaxis], y_te_aug), callbacks=[early_stop], verbose=0)
        s_scores['EEGNet'].append(eegnet.evaluate(X_te_dl[..., np.newaxis], y_te_aug, verbose=0)[1])
        
        # 1D-CNN
        tf.keras.backend.clear_session()
        cnn = build_1d_cnn(Chans=30, Samples=X_tr_dl.shape[2])
        cnn.fit(np.transpose(X_tr_dl, (0, 2, 1)), y_tr_aug, epochs=50, batch_size=32, validation_data=(np.transpose(X_te_dl, (0, 2, 1)), y_te_aug), callbacks=[early_stop], verbose=0)
        s_scores['CNN_1D'].append(cnn.evaluate(np.transpose(X_te_dl, (0, 2, 1)), y_te_aug, verbose=0)[1])

    # Save and Print
    for k in scores.keys(): scores[k].append(np.mean(s_scores[k]))
    print(f"Sub {sub} | CSP:{scores['CSP_LDA'][-1]*100:.1f}% | FBCSP:{scores['FBCSP'][-1]*100:.1f}% | FBTS:{scores['FBTS'][-1]*100:.1f}% | EEGNet:{scores['EEGNet'][-1]*100:.1f}%")
    
    pd.DataFrame({'Subject': subjects[:len(scores['CSP_LDA'])], **scores}).to_csv(csv_path, index=False)

print("\n" + "="*50 + "\n FINAL BENCHMARKS \n" + "="*50)
for k, v in scores.items(): print(f"{k.ljust(10)}: {np.mean(v)*100:.2f}%")
