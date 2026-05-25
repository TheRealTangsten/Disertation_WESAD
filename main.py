import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Importurile tale specifice proiectului
import constants as cnst
import data_loading as dataLoading
import single_model_utils as smu  # Utilitarele cu noile funcții mecanice


# 1. Setări de determinism global (pentru reproductibilitate)
def set_global_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_global_determinism(42)

# 2. Configurații și constante
DATA_PATH = cnst.path_data
ALL_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
TEST_SUBJECTS = ['S15', 'S16', 'S17']
class_names = ['Baseline', 'Stress', 'Amusement']


# Funcția grafică păstrată din scriptul original
def plot_subject_confusion_matrices(subject_id, y_true, y_pred_rf, y_pred_cnn, y_pred_trans, y_pred_lstm, classes):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Confusion Matrices for Subject {subject_id}', fontsize=16)

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


# -------------------- Pipeline Execuție --------------------

def main():
    print("Loading preprocessed data from JSON...")
    #######################################################################################################################################
    full_df = dataLoading.load_processed_data(json_type="chest", include_resp=True)
    chest_df_hrv = dataLoading.load_processed_data(json_type="chest", include_hrv=True, include_eda= False,  include_resp=False)
    chest_df_eda = dataLoading.load_processed_data(json_type="chest", include_hrv=False, include_eda= True,  include_resp=False)
    chest_df_resp = dataLoading.load_processed_data(json_type="chest", include_hrv=False, include_eda= False,  include_resp=True)

    # Encodăm etichetele text în valori numerice (0, 1, 2)
    le = LabelEncoder()
    full_df['Label'] = le.fit_transform(full_df['Label'])
    chest_df_hrv['Label'] = le.fit_transform(chest_df_hrv['Label'])
    chest_df_eda['Label'] = le.fit_transform(chest_df_eda['Label'])
    chest_df_resp['Label'] = le.fit_transform(chest_df_resp['Label'])
    num_classes = len(le.classes_)

    # SPLIT DATE: SUBIECȚI ANTRENARE vs SUBIECȚI TEST
    print(f"\n[INFO] Splitting Data. Test Subjects: {TEST_SUBJECTS}")
    test_data_all = full_df[full_df['Subject'].isin(TEST_SUBJECTS)].copy()
    train_data_all = full_df[~full_df['Subject'].isin(TEST_SUBJECTS)].copy()

    chest_hrv_test_data = chest_df_hrv[chest_df_hrv['Subject'].isin(TEST_SUBJECTS)].copy()
    chest_eda_test_data = chest_df_eda[chest_df_eda['Subject'].isin(TEST_SUBJECTS)].copy()
    chest_resp_test_data = chest_df_resp[chest_df_resp['Subject'].isin(TEST_SUBJECTS)].copy()

    chest_hrv_train_data = chest_df_hrv[~chest_df_hrv['Subject'].isin(TEST_SUBJECTS)].copy()
    chest_eda_train_data = chest_df_eda[~chest_df_eda['Subject'].isin(TEST_SUBJECTS)].copy()
    chest_resp_train_data = chest_df_resp[~chest_df_resp['Subject'].isin(TEST_SUBJECTS)].copy()

    # Separăm feature-urile de label/subiect pentru setul de train
    X_train = train_data_all.drop(columns=["Label", "Subject"])

    chest_hrv_X_train = chest_hrv_train_data.drop(columns=["Label","Subject"])
    chest_eda_X_train = chest_eda_train_data.drop(columns=["Label","Subject"])
    chest_resp_X_train = chest_resp_train_data.drop(columns=["Label","Subject"])


    y_train = train_data_all["Label"].values

    chest_hrv_Y_train = chest_hrv_train_data["Label"].values
    chest_eda_Y_train =  chest_eda_train_data["Label"].values
    chest_resp_Y_train = chest_resp_train_data["Label"].values

    concat_chest_X_train_hrv_eda_resp = []
    concat_chest_X_train_hrv_eda_resp.append(chest_hrv_X_train)
    concat_chest_X_train_hrv_eda_resp.append(chest_eda_X_train)
    concat_chest_X_train_hrv_eda_resp.append(chest_resp_X_train)

    concat_chest_Y_train_hrv_eda_resp = []
    concat_chest_Y_train_hrv_eda_resp.append(chest_hrv_Y_train)
    concat_chest_Y_train_hrv_eda_resp.append(chest_eda_Y_train)
    concat_chest_Y_train_hrv_eda_resp.append(chest_resp_Y_train)

    print(f"Training Data Size: {len(X_train)} samples")
    #######################################################################################################################################
    # --- ANTRENARE INDIVIDUALĂ CU NOILE FUNCTII ---
    models_to_train = ['RF', 'CNN', 'TRANS', 'LSTM']
    trained_models = {}

    for m_name in models_to_train:
        # Apelăm funcția train_model din noul fișier utilitar
        trained_models[m_name] = smu.train_model(X_train, y_train, num_classes, m_name)

    xt, yt, _ = dataLoading.provide_train_data_concat(option = "chest", hrv = True, eda = True, resp = True)
    #trained_multi_cnn = smu.train_multi_branch_by_vector_count(concat_chest_X_train_hrv_eda_resp, concat_chest_Y_train_hrv_eda_resp, num_classes)
    trained_multi_cnn = smu.train_multi_branch_by_vector_count(xt, yt, num_classes)
    # --- EVALUARE PE FIECARE SUBIECT DE TEST ---
    print("\n=== STARTING EVALUATION ON TEST SUBJECTS ===")
    results = []

    for sub_id in TEST_SUBJECTS:
        sub_data = test_data_all[test_data_all['Subject'] == sub_id]
        sub_chest_hrv_test_data = chest_hrv_test_data[chest_hrv_test_data['Subject'] == sub_id]
        sub_chest_eda_test_data = chest_eda_test_data[chest_eda_test_data['Subject'] == sub_id]
        sub_chest_resp_test_data = chest_resp_test_data[chest_resp_test_data['Subject'] == sub_id]
        if len(sub_data) == 0:
            continue

        X_test_sub = sub_data.drop(columns=["Label", "Subject"])
        X_sub_chest_hrv_test_data = sub_chest_hrv_test_data.drop(columns=["Label", "Subject"])
        X_sub_chest_eda_test_data = sub_chest_eda_test_data.drop(columns=["Label", "Subject"])
        X_sub_chest_resp_test_data = sub_chest_resp_test_data.drop(columns=["Label", "Subject"])
        concat_X_sub_chest_3 = []
        concat_X_sub_chest_3.append(X_sub_chest_hrv_test_data)
        concat_X_sub_chest_3.append(X_sub_chest_eda_test_data)
        concat_X_sub_chest_3.append(X_sub_chest_resp_test_data)

        y_test_sub = sub_data["Label"].values
        Y_sub_chest_hrv_test_data = sub_chest_hrv_test_data["Label"].values
        Y_sub_chest_eda_test_data = sub_chest_eda_test_data["Label"].values
        Y_sub_chest_resp_test_data = sub_chest_resp_test_data["Label"].values
        concat_Y_sub_chest_3 = []
        concat_Y_sub_chest_3.append(Y_sub_chest_hrv_test_data)
        concat_Y_sub_chest_3.append(Y_sub_chest_eda_test_data)
        concat_Y_sub_chest_3.append(Y_sub_chest_resp_test_data)

        # Obținem predicțiile și acuratețea pentru fiecare model în parte folosind predict_model
        acc_rf, y_pred_rf = smu.predict_model(trained_models['RF'], X_test_sub, y_test_sub, 'RF')
        acc_cnn, y_pred_cnn = smu.predict_model(trained_models['CNN'], X_test_sub, y_test_sub, 'CNN')
        acc_trans, y_pred_trans = smu.predict_model(trained_models['TRANS'], X_test_sub, y_test_sub, 'TRANS')
        acc_lstm, y_pred_lstm = smu.predict_model(trained_models['LSTM'], X_test_sub, y_test_sub, 'LSTM')

        xxt, yyt, _ = dataLoading.provide_test_data_concat(sub_id, option="chest", hrv = True, eda = True, resp = True)
        acc_multi_cnn_3, y_pred_multi_cnn_3 =  smu.predict_multi_branch_by_vector_count(trained_multi_cnn, xxt, yyt )

        print(
            f"\n  Result {sub_id}: RF={acc_rf:.2f}, CNN={acc_cnn:.2f}, Transformer={acc_trans:.2f}, LSTM={acc_lstm:.2f}")

        # Salvăm metricile obținute în listă
        results.append({
            'subject': sub_id,
            'acc_rf': acc_rf,
            'acc_cnn': acc_cnn,
            'acc_transformer': acc_trans,
            'acc_lstm': acc_lstm,
            'acc_multi_cnn_3': acc_multi_cnn_3,
        })

        # Afișarea matricelor de confuzie aferente subiectului curent
        print(f"  Displaying Confusion Matrices for {sub_id}...")
        plot_subject_confusion_matrices(
            sub_id,
            y_test_sub,
            y_pred_rf,
            y_pred_cnn,
            y_pred_trans,
            y_pred_lstm,
            class_names
        )

    # --- AFISARE REZULTATE FINALE MEDII ---
    df_results = pd.DataFrame(results)
    print("\n=== FINAL RESULTS ===")
    print(df_results.to_string(index=False))

    if not df_results.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"RF:  {df_results['acc_rf'].mean():.2f}")
        print(f"CNN: {df_results['acc_cnn'].mean():.2f}")
        print(f"Transformer (TRANS): {df_results['acc_transformer'].mean():.2f}")
        print(f"LSTM: {df_results['acc_lstm'].mean():.2f}")
        print(f"Multi CNN 3: {df_results['acc_multi_cnn_3'].mean():.2f}")


if __name__ == "__main__":
    main()