import constants as cnst # <<< !!! AJUSTATI PATH-UL CATRE SETUL DE DATE !!!
#import cnn_model as cnn
#import transformer_model as transformer
import use_model as model
import data_loading as dataLoading


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
#TEST_SUBJECTS_2 = ['S11', 'S13', 'S14', 'S15', 'S16', 'S17']

SAMPLING_RATE = 700
WINDOW_SIZE_SEC = 120
WINDOW_STEP_SEC = 40
window_size_samples = WINDOW_SIZE_SEC * SAMPLING_RATE
step_size_samples = WINDOW_STEP_SEC * SAMPLING_RATE


"""
def prepare_global_dataset(all_ids, source = "chest"):
    all_data_frames = []
    print("\n--- START GLOBAL DATA EXTRACTION ---")
    for sub_id in all_ids:
        if source == "chest":
            df_sub = extract_features_from_subject(sub_id)
        else: # wrist data
            df_sub = extract_wrist_features_from_subject(sub_id)
        if df_sub is not None:
            all_data_frames.append(df_sub)

    if not all_data_frames:
        raise ValueError("No data loaded!")

    full_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"--- DATA LOADED: {full_df.shape} samples ---")
    return full_df
"""

def plot_subject_confusion_matrices(subject_id, y_true, y_pred_rf, y_pred_cnn, y_pred_trans, y_pred_lstm, classes):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Confusion Matrices for Subject {subject_id}', fontsize=16)

    #model_names = ['Random Forest', 'CNN', 'Transformer']
    model_names = ['Random Forest', 'CNN', 'Transformer', 'LSTM']
    predictions = [y_pred_rf, y_pred_cnn, y_pred_trans, y_pred_lstm]

    axes = axes.flatten()

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
#full_df = prepare_global_dataset(ALL_SUBJECTS, source='chest')
# Apelăm funcția; va salva fișierele în folderul curent
#dataLoading.preprocess_and_save_to_json(ALL_SUBJECTS)

print("Loading preprocessed data from JSON...")
#full_df = pd.read_json("Jsons\chest.json", orient="records")
full_df = dataLoading.load_processed_data(json_type="chest", include_resp=False)


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

##
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
##

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

    ##
    #X_test_sub_scaled = scaler.transform(X_test_sub)
    #X_test_sub = pd.DataFrame(X_test_sub_scaled, columns=X_test_sub.columns)
    ##

    # Predictie
    res = model.predict_on_test_data(trained_models, X_test_sub, y_test_sub)

    print(f"X_Test: {X_test_sub}\nY_Test: {y_test_sub}")
    print(f"  Result {sub_id}: RF={res['acc_rf']:.2f}, CNN={res['acc_cnn']:.2f}, Transformer={res['acc_transformer']:.2f}, LSTM={res['acc_lstm']:.2f}")

    # Construire lista rezultate
    results.append({
        'subject': sub_id,
        'acc_rf': res['acc_rf'],
        'acc_cnn': res['acc_cnn'],
        'acc_transformer': res['acc_transformer'],
        'acc_lstm': res['acc_lstm']
    })

    # --- AFISARE MATRICE DE CONFUZIE ---
    print(f"  Displaying Confusion Matrices for {sub_id}...")
    plot_subject_confusion_matrices(
        sub_id,
        y_test_sub,
        res['y_pred_rf'],
        res['y_pred_cnn'],
        res['y_pred_trans'],
        res['y_pred_lstm'],
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
    print(f"LSTM: {df_results['acc_lstm'].mean():.2f}")