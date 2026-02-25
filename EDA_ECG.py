import pickle
import constants as cnst
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit  # MODIFICARE: Pentru a rezolva leakage-ul
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk

# Config
DATA_PATH = cnst.path_data
subject_ids = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
SAMPLING_RATE = 700  # WESAD Chest data is 700Hz for both ECG and EDA
WINDOW_SIZE_SEC = 120
WINDOW_STEP_SEC = 60
window_size_samples = WINDOW_SIZE_SEC * SAMPLING_RATE
step_size_samples = WINDOW_STEP_SEC * SAMPLING_RATE

# Store all detailed results
all_results = []


def process_subject(subject_id):
    print(f"\n=== Processing {subject_id} ===")
    try:
        with open(f"{DATA_PATH}{subject_id}/{subject_id}.pkl", 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Could not load data for {subject_id}: {e}")
        return None

    # --- 1. PRELUARE SEMNALE ---
    ecg_signal = data['signal']['chest']['ECG']
    eda_signal = data['signal']['chest']['EDA'].flatten()  # MODIFICARE: Preluare EDA
    labels = data['label']

    # --- 2. PROCESARE ECG ---
    # Clean ECG & find R-peaks
    cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=SAMPLING_RATE)
    _, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=SAMPLING_RATE)
    rpeaks_indices = rpeaks['ECG_R_Peaks']

    # --- 3. PROCESARE EDA (Optimizat) ---
    # Procesăm tot semnalul o singură dată pentru a extrage Tonic și Phasic component
    # Aceasta metoda este mult mai rapidă decât procesarea în buclă
    print("  Processing EDA signal...")
    try:
        eda_processed, _ = nk.eda_process(eda_signal, sampling_rate=SAMPLING_RATE)
        # eda_processed conține coloane precum: 'EDA_Clean', 'EDA_Tonic', 'EDA_Phasic'
    except Exception as e:
        print(f"  EDA processing failed: {e}")
        return None

    # --- 4. SLIDING WINDOW & FUSION ---
    print("  Extracting features...")
    features_list = []

    # Iterăm prin semnal
    for start in range(0, len(ecg_signal) - window_size_samples, step_size_samples):
        end = start + window_size_samples

        # a) Label processing
        window_labels = labels[start:end]
        most_common_label = np.bincount(window_labels).argmax()

        # Filtru rapid: Ignorăm ferestrele care nu sunt Baseline(1), Stress(2), Amusement(3)
        if most_common_label not in [1, 2, 3]:
            continue

        # b) ECG Features (HRV)
        peaks_in_window = rpeaks_indices[(rpeaks_indices >= start) & (rpeaks_indices < end)] - start

        if len(peaks_in_window) > 2:
            try:
                peaks_df = pd.DataFrame({"ECG_R_Peaks": np.zeros(window_size_samples, dtype=bool)})
                peaks_df.loc[peaks_in_window, "ECG_R_Peaks"] = True

                # Extrage HRV
                hrv = nk.hrv(peaks_df, sampling_rate=SAMPLING_RATE, show=False)
                hrv_row = hrv.iloc[0].to_dict()  # Convertim la dict

                # c) EDA Features (Statistical features on segments)
                # Extragem segmentul corespunzător din datele EDA procesate anterior
                eda_window = eda_processed.iloc[start:end]

                eda_feats = {
                    'EDA_Mean': eda_window['EDA_Clean'].mean(),
                    'EDA_Std': eda_window['EDA_Clean'].std(),
                    'EDA_Tonic_Mean': eda_window['EDA_Tonic'].mean(),
                    'EDA_Tonic_Std': eda_window['EDA_Tonic'].std(),
                    'EDA_Phasic_Mean': eda_window['EDA_Phasic'].mean(),
                    'EDA_Phasic_Max': eda_window['EDA_Phasic'].max(),
                    'EDA_Phasic_Std': eda_window['EDA_Phasic'].std()
                }

                # d) FUSION: Combinăm dicționarele
                fused_row = {**hrv_row, **eda_feats}
                fused_row["Label"] = most_common_label

                features_list.append(fused_row)

            except Exception as e:
                continue

    if not features_list:
        print(f"No features extracted for {subject_id}")
        return None

    df_features = pd.DataFrame(features_list)

    # Curățare NaN-uri
    df_features = df_features.fillna(df_features.median())
    # Eliminam coloane cu prea multe NaN (dacă există)
    nan_counts = df_features.isna().sum()
    cols_to_drop = nan_counts[nan_counts > (0.3 * len(df_features))].index
    df_features = df_features.drop(columns=cols_to_drop)

    # Pregătire date
    X = df_features.drop(columns=["Label"])
    y = df_features["Label"]

    if len(y.unique()) < 2 or len(y) < 10:
        print(f"Not enough data to train for {subject_id}")
        return None

    # --- 5. REZOLVARE DATA LEAKAGE (Group Split) ---
    # Standardizare
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Păstrăm ca DF pentru consistență

    # Definire Grupuri:
    # Deoarece Step = Window / 2, avem overlap de 50%.
    # Fereastra i și i+1 împart date. Le punem în același grup.
    # groups = [0, 0, 1, 1, 2, 2, ...]
    groups = np.arange(len(y)) // 2

    # Folosim GroupShuffleSplit în loc de train_test_split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Split
    for train_idx, test_idx in gss.split(X_scaled, y, groups):
        X_train, y_train = X_scaled.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X_scaled.iloc[test_idx], y.iloc[test_idx]

    # --- 6. MODELING ---
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])

    return {
        'subject': subject_id,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'n_samples': len(y),
        'confusion_matrix': cm
    }


# Process all subjects
for subject_id in subject_ids:
    res = process_subject(subject_id)
    if res:
        all_results.append(res)

# Build summary DataFrame
df_results = pd.DataFrame([{
    'subject': r['subject'],
    'accuracy': r['accuracy'],
    'macro_f1': r['macro_f1'],
    'n_samples': r['n_samples']
} for r in all_results]).sort_values(by='accuracy', ascending=False)

print("\n=== Summary Table (sorted by accuracy) ===")
print(df_results)

# Mean ± std
mean_acc = df_results['accuracy'].mean()
std_acc = df_results['accuracy'].std()
mean_f1 = df_results['macro_f1'].mean()
std_f1 = df_results['macro_f1'].std()

print(f"\nMean accuracy: {mean_acc:.2f} ± {std_acc:.2f}")
print(f"Mean macro F1: {mean_f1:.2f} ± {std_f1:.2f}")

# Plot: average normalized confusion matrix
cms = [r['confusion_matrix'] for r in all_results]
sum_cm = np.sum(cms, axis=0)
norm_cm = sum_cm / sum_cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))
sns.heatmap(norm_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
plt.title('Average normalized confusion matrix (ECG + EDA)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

"""
=== Summary Table (sorted by accuracy) ===
   subject  accuracy  macro_f1  n_samples
1       S3     1.000  1.000000         37
2       S4     1.000  1.000000         36
14     S17     1.000  1.000000         38
11     S14     1.000  1.000000         38
9      S11     1.000  1.000000         38
8      S10     1.000  1.000000         38
7       S9     1.000  1.000000         36
13     S16     1.000  1.000000         37
0       S2     0.875  0.466667         36
5       S7     0.875  0.636364         38
3       S5     0.875  0.466667         36
10     S13     0.875  0.794872         38
4       S6     0.750  0.600000         37
6       S8     0.750  0.733333         38
12     S15     0.750  0.555556         39
"""