import constants as const
import os

def show_all_files():
    for dirname, _, filenames in os.walk( const.path_data ):
        for filename in filenames:
            print(os.path.join(dirname, filename))


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk # Excellent for biosignal processing
import pywt # For wavelet transforms
import seaborn as sns

show_all_files()

DATA_PATH = const.path_data
SUBJECT_ID = 'S2' # Example subject ID to process
ECG_SAMPLING_RATE_RESP = 700 # Hz (from WESAD documentation for RespiBAN ECG)
LABEL_SAMPLING_RATE = 700 # Hz (from WESAD documentation for labels)
HRV_INTERPOLATION_RATE = 4 # Hz (Common for HRV spectral analysis, as per RHRV example in the wavelet paper)
WINDOW_SIZE_SEC = 300 # 5 minutes for rolling feature aggregation
WINDOW_OVERLAP_SEC = 60 # 1 minute overlap for smooth features

# Set a style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('deep')

print(f"--- Preprocessing for Subject: {SUBJECT_ID} ---")

# 1. Load Data
print(f"Loading data for {SUBJECT_ID} from {DATA_PATH}{SUBJECT_ID}/{SUBJECT_ID}.pkl...")
try:
    with open(f"{DATA_PATH}{SUBJECT_ID}/{SUBJECT_ID}.pkl", 'rb') as f:
        data = pickle.load(f, encoding='latin1') # Use latin1 for Python 3 compatibility with older pickle files
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: {DATA_PATH}{SUBJECT_ID}/{SUBJECT_ID}.pkl not found. Please check DATA_PATH.")
    exit()

print("\n--- Data Structure Overview ---")
print(f"Keys in loaded data: {data.keys()}")

signal_data = data['signal']
print(f"Keys in signal data: {signal_data.keys()}")

chest_data = signal_data['chest']
wrist_data = signal_data['wrist']
print(f"Keys in chest data: {chest_data.keys()}")
print(f"Keys in wrist data: {wrist_data.keys()}")
print(f"Labels: \n {data['label']}")

ecg_signal = data['signal']['chest']['ECG']
eda_signal = data['signal']['chest']['EDA']  # MODIFICARE: Preluare EDA

labels = data['label']
unique_labels = set(labels)
dict_labels = {0:0}
for i in unique_labels:
    dict_labels[i] = 0
for i in labels:
    dict_labels[i] = dict_labels[i] + 1
print(f"{dict_labels.keys()} - {unique_labels}\n{dict_labels}\n\n")



try:
    # Clean ECG signal
    cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=ECG_SAMPLING_RATE_RESP)

    # Detect R-peaks
    _, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=ECG_SAMPLING_RATE_RESP)

    # Extract R-peak indices
    rpeaks_indices = rpeaks['ECG_R_Peaks']

    print(f"Detected {len(rpeaks_indices)} R-peaks.")

    if len(rpeaks_indices) > 1:
        # Compute RR intervals in ms
        rr_intervals_ms = np.diff(rpeaks_indices) / ECG_SAMPLING_RATE_RESP * 1000
        rr_times_sec = rpeaks_indices[1:] / ECG_SAMPLING_RATE_RESP

        rr_df = pd.DataFrame({'Time_sec': rr_times_sec, 'RR_ms': rr_intervals_ms})

        print(f"Calculated {len(rr_intervals_ms)} RR intervals.")
        print(f"First 5 RR intervals:\n{rr_df.head()}")
    else:
        print("Too few R-peaks to compute RR intervals.")

except Exception as e:
    print(f"An error occurred: {e}")

print(rpeaks.keys())

print(f"ECG: {ecg_signal}")
print(f"EDA: {eda_signal}")
