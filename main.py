

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import constants as const
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

def show_all_files():
    for dirname, _, filenames in os.walk( const.path_data ):
        for filename in filenames:
            print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk # Excellent for biosignal processing
import pywt # For wavelet transforms
import seaborn as sns

show_all_files()
# --- Configuration ---
# IMPORTANT: Adjust this path to where your WESAD dataset is located on Kaggle
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
print(f"Keys in chest data: {chest_data.keys()}")

# Extract ECG and labels
ecg_signal = chest_data['ECG']
labels = data['label']

print(f"ECG signal shape: {ecg_signal.shape}")
print(f"Labels shape: {labels.shape}")

print("\n--- Data Structure Overview ---")
print(f"Keys in loaded data: {data.keys()}")

signal_data = data['signal']
print(f"Keys in signal data: {signal_data.keys()}")

chest_data = signal_data['chest']
print(f"Keys in chest data: {chest_data.keys()}")

# Extract ECG and labels
ecg_signal = chest_data['ECG']
labels = data['label']

print(f"ECG signal shape: {ecg_signal.shape}")
print(f"Labels shape: {labels.shape}")

plt.figure(figsize=(15, 4))
plt.plot(np.arange(len(ecg_signal[:ECG_SAMPLING_RATE_RESP*10])) / ECG_SAMPLING_RATE_RESP,
         ecg_signal[:ECG_SAMPLING_RATE_RESP*10])
plt.title(f'Raw ECG Signal (first 10 seconds) - Subject {SUBJECT_ID}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.figure(figsize=(15, 3))
plt.plot(np.arange(len(labels)) / LABEL_SAMPLING_RATE, labels, alpha=0.7)
plt.title('Study protocol labels')
plt.xlabel('Time (s)')
plt.ylabel('Label')
plt.show()

############# ---------------------------------------------

print("\n--- II. RR Interval Extraction from ECG ---")

print(f"Processing ECG signal of length {len(ecg_signal)} samples at {ECG_SAMPLING_RATE_RESP} Hz...")

# Initialize variables to safe defaults
rr_df = pd.DataFrame()
rpeaks_indices = np.array([])

try:
    # Clean ECG signal
    print("Cleaning ECG signal...")
    cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=ECG_SAMPLING_RATE_RESP)

    # Detect R-peaks
    print("Detecting R-peaks...")
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



plt.figure(figsize=(12,4))
plt.plot(rr_df['Time_sec'], rr_df['RR_ms'])
plt.title(f'RR intervals over time - Subject {SUBJECT_ID}')
plt.xlabel('Time (s)')
plt.ylabel('RR interval (ms)')
plt.show()

############# ---------------------------------------------


print("\n--- Sliding window HRV feature extraction ---")

WINDOW_SIZE_SEC = 120  # try 2 minutes
WINDOW_STEP_SEC = 60  # 1-minute overlap
window_size_samples = WINDOW_SIZE_SEC * ECG_SAMPLING_RATE_RESP
step_size_samples = WINDOW_STEP_SEC * ECG_SAMPLING_RATE_RESP

print(f"Window size: {window_size_samples} samples")
print(f"Step size: {step_size_samples} samples")

hrv_features_list = []

for start in range(0, len(ecg_signal) - window_size_samples, step_size_samples):
    end = start + window_size_samples
    window_center_sec = (start + end) / 2 / ECG_SAMPLING_RATE_RESP

    # Find R-peaks in this window
    peaks_in_window = rpeaks_indices[(rpeaks_indices >= start) & (rpeaks_indices < end)] - start

    print(f"Window starting at {start}: {len(peaks_in_window)} peaks")

    if len(peaks_in_window) > 2:
        # Build peaks DataFrame expected by nk.hrv
        peaks_df = pd.DataFrame({"ECG_R_Peaks": np.zeros(window_size_samples, dtype=bool)})
        peaks_df.loc[peaks_in_window, "ECG_R_Peaks"] = True

        try:
            hrv = nk.hrv(peaks_df, sampling_rate=ECG_SAMPLING_RATE_RESP, show=False)
            hrv_row = hrv.iloc[0].to_dict()

            # Add time and label
            window_labels = labels[start:end]
            most_common_label = np.bincount(window_labels).argmax()
            hrv_row["Time"] = window_center_sec
            hrv_row["Label"] = most_common_label

            hrv_features_list.append(hrv_row)

        except Exception as e:
            print(f"Error computing HRV in window starting at {start}: {e}")
    else:
        print(f"Too few peaks in window starting at {start}, skipping.")

# Convert to DataFrame
df_features = pd.DataFrame(hrv_features_list)
print("\n--- HRV Feature Table ---")
print("Shape:", df_features.shape)
print("Columns:", df_features.columns)
if not df_features.empty:
    print("Unique labels:", df_features["Label"].unique())
    print(df_features.head())
else:
    print("No HRV features extracted.")



print("\n--- HRV Feature Table ---")
print("Shape:", df_features.shape)
print("Columns:", df_features.columns)
print("Unique labels:", df_features["Label"].unique())
print(df_features.head())


print(f"Window starting at {start}: {len(peaks_in_window)} peaks")


plt.figure(figsize=(12,4))
plt.plot(df_features['Time'], df_features['HRV_SDNN'])
plt.title(f'HRV_SDNN over time - Subject {SUBJECT_ID}')
plt.xlabel('Time (s)')
plt.ylabel('HRV_SDNN')
plt.show()

############# ---------------------------------------------
df_features["Label"] = df_features["Label"].map(lambda x: x if x in [0,1,2,3] else 0)

df_features = df_features[df_features["Label"].isin([0,1,2,3])]

print(df_features["Label"].value_counts())


df_features = df_features.fillna(df_features.median())

nan_counts = df_features.isna().sum()
df_features = df_features.drop(columns=nan_counts[nan_counts > (0.2*len(df_features))].index)

X = df_features.drop(columns=["Time", "Label"])
y = df_features["Label"]

print("Final X shape:", X.shape)
print("Final y distribution:\n", y.value_counts())


############# ---------------------------------------------

print("Number of columns dropped due to >20% NaN:", len(nan_counts[nan_counts > (0.2*len(df_features))]))
print("Remaining feature columns:", X.shape[1])

############# ---------------------------------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

############# ---------------------------------------------

print("Test accuracy:", clf.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_scaled, y, cv=5)
print("Cross-validation accuracy: %.2f ± %.2f" % (scores.mean(), scores.std()))

import matplotlib.pyplot as plt
import numpy as np

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 10

plt.figure(figsize=(10,6))
plt.title("Top Feature Importances")
plt.bar(range(top_n), importances[indices[:top_n]], align='center')
plt.xticks(range(top_n), np.array(X.columns)[indices[:top_n]], rotation=45)
plt.tight_layout()
plt.show()





