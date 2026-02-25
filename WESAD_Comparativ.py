import constants as cnst # <<< !!! AJUSTATI PATH-UL CATRE SETUL DE DATE !!!
#import cnn_model as cnn
#import transformer_model as transformer
import use_model as model


import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

import neurokit2 as nk
import os
import random

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
#from keras.models import Sequential, Model
#from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, MultiHeadAttention, LayerNormalization, \
#    Add, GlobalAveragePooling1D
#from keras.utils import to_categorical
#from keras.optimizers import Adam


# setari seeds global
def set_global_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_global_determinism(42)

# Config
DATA_PATH = cnst.path_data
# Lista subiecti
ALL_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
TEST_SUBJECTS = ['S15', 'S16', 'S17']

SAMPLING_RATE = 700
WINDOW_SIZE_SEC = 120
WINDOW_STEP_SEC = 40
window_size_samples = WINDOW_SIZE_SEC * SAMPLING_RATE
step_size_samples = WINDOW_STEP_SEC * SAMPLING_RATE

def extract_features_from_subject(subject_id):
    print(f"Loading and normalizing data for {subject_id}...")
    try:
        with open(f"{DATA_PATH}{subject_id}/{subject_id}.pkl", 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Could not load data for {subject_id}: {e}")
        return None

    ecg_signal = data['signal']['chest']['ECG']
    eda_signal = data['signal']['chest']['EDA'].flatten()
    labels = data['label']

    try:
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=SAMPLING_RATE)
        _, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=SAMPLING_RATE)
        rpeaks_indices = rpeaks['ECG_R_Peaks']
        eda_processed, _ = nk.eda_process(eda_signal, sampling_rate=SAMPLING_RATE)
    except Exception as e:
        print(f"  Signal processing failed for {subject_id}: {e}")
        return None

    features_list = []

    # Iteratie ferestre
    for start in range(0, len(ecg_signal) - window_size_samples, step_size_samples):
        end = start + window_size_samples

        if end > len(labels): break

        window_labels = labels[start:end]
        most_common_label = np.bincount(window_labels).argmax()

        # 1=Baseline, 2=Stress, 3=Amusement, excludere rest
        if most_common_label not in [1, 2, 3]:
            continue

        peaks_in_window = rpeaks_indices[(rpeaks_indices >= start) & (rpeaks_indices < end)] - start

        # Minim 3 batai de inima pentru HRV
        if len(peaks_in_window) > 3:
            try:
                # 1. HRV Features
                peaks_df = pd.DataFrame({"ECG_R_Peaks": np.zeros(window_size_samples, dtype=bool)})
                peaks_df.loc[peaks_in_window, "ECG_R_Peaks"] = True

                hrv = nk.hrv(peaks_df, sampling_rate=SAMPLING_RATE, show=False)
                # selectare doar numere
                hrv_numeric = hrv.select_dtypes(include=[np.number])
                hrv_row = hrv_numeric.iloc[0].to_dict()

                # 2. EDA Features
                eda_window = eda_processed.iloc[start:end]
                eda_feats = {
                    'EDA_Mean': eda_window['EDA_Clean'].mean(),
                    'EDA_Std': eda_window['EDA_Clean'].std(),
                    'EDA_Tonic_Mean': eda_window['EDA_Tonic'].mean(),
                    'EDA_Phasic_Mean': eda_window['EDA_Phasic'].mean(),
                    'EDA_Phasic_Std': eda_window['EDA_Phasic'].std(),
                    'EDA_Min': eda_window['EDA_Clean'].min(),
                    'EDA_Max': eda_window['EDA_Clean'].max()
                }

                fused_row = {**hrv_row, **eda_feats}
                fused_row["Label"] = most_common_label
                fused_row["Subject"] = subject_id
                features_list.append(fused_row)

            except Exception:
                continue

    if not features_list:
        return None

    df = pd.DataFrame(features_list)

    # handling valori invalide

    # valori infinite inlocuite cu NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Inlocuire valori NaN cu mediana ( daca exista )
    df = df.fillna(df.median(numeric_only=True))
    # Inlocuire colana NaN cu 0
    df = df.fillna(0)

    #Normalizare per-subiect
    feature_cols = [c for c in df.columns if c not in ['Label', 'Subject']]

    if not feature_cols:
        print(f"  Warning: No valid feature columns for {subject_id}")
        return None

    scaler = StandardScaler()
    try:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        df[feature_cols] = df[feature_cols].fillna(0)
    except ValueError as e:
        print(f"  Scaling error for {subject_id}: {e}")
        return None

    return df


def prepare_global_dataset(all_ids):
    all_data_frames = []
    print("\n--- START GLOBAL DATA EXTRACTION ---")
    for sub_id in all_ids:
        df_sub = extract_features_from_subject(sub_id)
        if df_sub is not None:
            all_data_frames.append(df_sub)

    if not all_data_frames:
        raise ValueError("No data loaded!")

    full_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"--- DATA LOADED: {full_df.shape} samples ---")
    return full_df


def plot_subject_confusion_matrices(subject_id, y_true, y_pred_rf, y_pred_cnn, y_pred_trans, classes):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'Confusion Matrices for Subject {subject_id}', fontsize=16)

    model_names = ['Random Forest', 'CNN', 'Transformer']
    predictions = [y_pred_rf, y_pred_cnn, y_pred_trans]

    for i, ax in enumerate(axes):
        cm = confusion_matrix(y_true, predictions[i])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=classes, yticklabels=classes)

        ax.set_title(f'{model_names[i]}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.show()


# -------------------- Main --------------------

# Data
full_df = prepare_global_dataset(ALL_SUBJECTS)

# Labels
le = LabelEncoder()
full_df['Label'] = le.fit_transform(full_df['Label'])
num_classes = len(le.classes_)

# SPLIT TRAIN / TEST
print(f"\n[INFO] Splitting Data. Test Subjects: {TEST_SUBJECTS}")

test_data_all = full_df[full_df['Subject'].isin(TEST_SUBJECTS)].copy()
train_data_all = full_df[~full_df['Subject'].isin(TEST_SUBJECTS)].copy() #excludere subiecti de test

X_train = train_data_all.drop(columns=["Label", "Subject"])
y_train = train_data_all["Label"].values

print(f"Training Data Size: {len(X_train)} samples")

trained_models = model.train_all_models_once(X_train, y_train, num_classes)

class_names = ['Baseline', 'Stress', 'Amusement']

print("\n=== STARTING EVALUATION ON TEST SUBJECTS ===")
results = []

for sub_id in TEST_SUBJECTS:
    sub_data = test_data_all[test_data_all['Subject'] == sub_id]
    if len(sub_data) == 0: continue

    X_test_sub = sub_data.drop(columns=["Label", "Subject"])
    y_test_sub = sub_data["Label"].values

    # Predictie
    res = model.predict_on_test_data(trained_models, X_test_sub, y_test_sub)

    print(f"  Result {sub_id}: RF={res['acc_rf']:.2f}, CNN={res['acc_cnn']:.2f}, Transformer={res['acc_transformer']:.2f}")

    # Construire lista rezultate
    results.append({
        'subject': sub_id,
        'acc_rf': res['acc_rf'],
        'acc_cnn': res['acc_cnn'],
        'acc_transformer': res['acc_transformer']
    })

    # --- AFISARE MATRICE DE CONFUZIE ---
    print(f"  Displaying Confusion Matrices for {sub_id}...")
    plot_subject_confusion_matrices(
        sub_id,
        y_test_sub,
        res['y_pred_rf'],
        res['y_pred_cnn'],
        res['y_pred_trans'],
        class_names
    )

#  REZULTATE FINALE
df_results = pd.DataFrame(results)
print("\n=== FINAL RESULTS ===")
print(df_results)
if not df_results.empty:
    print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
    print(f"RF: {df_results['acc_rf'].mean():.2f}")
    print(f"CNN: {df_results['acc_cnn'].mean():.2f}")
    print(f"Transformer: {df_results['acc_transformer'].mean():.2f}")