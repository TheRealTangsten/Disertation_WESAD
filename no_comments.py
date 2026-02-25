import pickle
import constants as cnst
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk

# Config
DATA_PATH = cnst.path_data
subject_ids = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
ECG_SAMPLING_RATE_RESP = 700
WINDOW_SIZE_SEC = 120
WINDOW_STEP_SEC = 60
window_size_samples = WINDOW_SIZE_SEC * ECG_SAMPLING_RATE_RESP
step_size_samples = WINDOW_STEP_SEC * ECG_SAMPLING_RATE_RESP

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

    # Extract ECG and labels
    ecg_signal = data['signal']['chest']['ECG']
    labels = data['label']

    # Clean ECG & find R-peaks
    cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=ECG_SAMPLING_RATE_RESP)
    _, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=ECG_SAMPLING_RATE_RESP)
    rpeaks_indices = rpeaks['ECG_R_Peaks']

    # Sliding window HRV features
    hrv_features_list = []
    for start in range(0, len(ecg_signal) - window_size_samples, step_size_samples):
        end = start + window_size_samples
        window_center_sec = (start + end) / 2 / ECG_SAMPLING_RATE_RESP

        peaks_in_window = rpeaks_indices[(rpeaks_indices >= start) & (rpeaks_indices < end)] - start

        if len(peaks_in_window) > 2:
            peaks_df = pd.DataFrame({"ECG_R_Peaks": np.zeros(window_size_samples, dtype=bool)})
            peaks_df.loc[peaks_in_window, "ECG_R_Peaks"] = True
            try:
                hrv = nk.hrv(peaks_df, sampling_rate=ECG_SAMPLING_RATE_RESP, show=False)
                hrv_row = hrv.iloc[0].to_dict()
                window_labels = labels[start:end]
                most_common_label = np.bincount(window_labels).argmax()
                hrv_row["Time"] = window_center_sec
                hrv_row["Label"] = most_common_label
                hrv_features_list.append(hrv_row)
            except Exception as e:
                continue

    if not hrv_features_list:
        print(f"No HRV features extracted for {subject_id}")
        return None

    df_features = pd.DataFrame(hrv_features_list)

    #df_features["Label"] = df_features["Label"].map(lambda x: x if x in [0, 1, 2, 3] else 0)
    #df_features["Label"] = df_features["Label"].map(lambda x: x if x in [1, 2, 3, 4] else 0)
    df_features["Label"] = df_features["Label"].map(lambda x: x if x in [1, 2, 3] else 0)
    #df_features = df_features[df_features["Label"].isin([0, 1, 2, 3])]
    #df_features = df_features[df_features["Label"].isin([1, 2, 3, 4])]
    df_features = df_features[df_features["Label"].isin([1, 2, 3])]

    df_features = df_features.fillna(df_features.median())
    nan_counts = df_features.isna().sum()
    df_features = df_features.drop(columns=nan_counts[nan_counts > (0.2 * len(df_features))].index)

    X = df_features.drop(columns=["Time", "Label"])
    y = df_features["Label"]

    if len(y.unique()) < 2 or len(y) < 10:
        print(f"Not enough data to train for {subject_id}")
        return None

    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    #cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    #cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
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

# Plot: per-subject confusion matrices
for r in all_results:
    plt.figure(figsize=(4, 4))
    #sns.heatmap(r['confusion_matrix'], annot=True, fmt='d', cmap='Blues',xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
    #sns.heatmap(r['confusion_matrix'], annot=True, fmt='d', cmap='Blues',xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
    sns.heatmap(r['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3],yticklabels=[1, 2, 3])
    plt.title(f"Confusion matrix: {r['subject']}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plot: average normalized confusion matrix
cms = [r['confusion_matrix'] for r in all_results]
sum_cm = np.sum(cms, axis=0)
norm_cm = sum_cm / sum_cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))
#sns.heatmap(norm_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
#sns.heatmap(norm_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
sns.heatmap(norm_cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
plt.title('Average normalized confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plot: barplots of accuracy and macro F1
plt.figure(figsize=(10, 5))
sns.barplot(x='subject', y='accuracy', data=df_results, order=df_results['subject'])
plt.ylim(0, 1)
plt.title('Accuracy per subject')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='subject', y='macro_f1', data=df_results, order=df_results['subject'])
plt.ylim(0, 1)
plt.title('Macro F1 per subject')
plt.ylabel('Macro F1')
plt.show()

# Print mean ± std
mean_acc = df_results['accuracy'].mean()
std_acc = df_results['accuracy'].std()
mean_f1 = df_results['macro_f1'].mean()
std_f1 = df_results['macro_f1'].std()

print(f"\nMean accuracy: {mean_acc:.2f} ± {std_acc:.2f}")
print(f"Mean macro F1: {mean_f1:.2f} ± {std_f1:.2f}")