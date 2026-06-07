import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import plotting as plting

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
class_names_binary = ['No Stress', 'Stress']

notation_cf = "Chest - HRV, EDA, RESP"
notation_che = "Chest - HRV, EDA"
notation_w = "Wrist - HRV, EDA"

# Funcția grafică păstrată din scriptul original
def plot_subject_confusion_matrices(subject_id, y_true, y_pred_rf, y_pred_cnn, y_pred_trans, y_pred_lstm, classes,
                                    model_names=['Random Forest', 'CNN', 'Transformer', 'LSTM']):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Confusion Matrices for Subject {subject_id}', fontsize=16)

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
    num_classes_3 = 3
    num_classes_2 = 2
    #######################################################################################################################################
    x3_train_chest_hrv_eda, y3_train_chest_hrv_eda, num_cls = dataLoading.provide_train_data_fused(option="chest",
                                                                                                   hrv=True, eda=True,
                                                                                                   resp=False)

    x3_train_chest_full, y3_train_chest_full, _ = dataLoading.provide_train_data_fused(option="chest", hrv=True,
                                                                                       eda=True, resp=True)

    x3_train_chest_hrv, y3_train_chest_hrv, _ = dataLoading.provide_train_data_fused(option="chest", hrv=True,
                                                                                     eda=False, resp=False)
    x3_train_chest_eda, y3_train_chest_eda, _ = dataLoading.provide_train_data_fused(option="chest", hrv=False,
                                                                                     eda=True, resp=False)
    x3_train_chest_resp, y3_train_chest_resp, _ = dataLoading.provide_train_data_fused(option="chest", hrv=False,
                                                                                       eda=False, resp=True)

    x3_train_wrist_full, y3_train_wrist_full, _ = dataLoading.provide_train_data_fused(option="wrist", hrv=True,
                                                                                       eda=True)

    x3_train_wrist_hrv, y3_train_wrist_hrv, _ = dataLoading.provide_train_data_fused(option="wrist", hrv=True,
                                                                                     eda=False)
    x3_train_wrist_eda, y3_train_wrist_eda, _ = dataLoading.provide_train_data_fused(option="wrist", hrv=False,
                                                                                     eda=True)

    print(f"Training Data Size: {len(x3_train_chest_hrv_eda)} samples")
    #######################################################################################################################################
    models_to_train = ['RF', 'CNN', 'TRANS', 'LSTM']
    trained_models_chest_hrv_eda = {}
    trained_models_chest_full = {}
    trained_models_wrist_full = {}

    # Classic Train: RF, CNN, TRANS, LSTM
    for m_name in models_to_train:
        trained_models_chest_full[m_name] = smu.train_model(x3_train_chest_full, y3_train_chest_full, num_classes_3,
                                                            m_name)



    ###### ---------------------  ######   --------------------- ######

    # --- EVALUARE PE FIECARE SUBIECT DE TEST ---
    print("\n=== STARTING EVALUATION ON TEST SUBJECTS ===")
    results_3cls_chest_hrv_eda = []
    results_3cls_chest_full = []
    results_3cls_wrist_full = []

    results_model_fusion_chest_full = []
    results_model_fusion_chest_hrv_eda = []
    results_model_fusion_wrist = []

    results_decision_fusion = []
    results_decision_fusion_2 = []

    results_2cls_chest_hrv_eda = []
    results_2cls_chest_full = []
    results_2cls_wrist_full = []

    results_fdf_chest_full = []
    results_fdf_chest_hrv_eda = []
    results_fdf_wrist = []

    for sub_id in TEST_SUBJECTS:
        _, y_test_sub_chest_3_cls = dataLoading.provide_test_data_fused(sub_id=sub_id, option="chest")
        _, y_test_sub_wrist_3_cls = dataLoading.provide_test_data_fused(sub_id=sub_id, option="wrist")
        _, y_test_sub_chest_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="chest")
        _, y_test_sub_wrist_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="wrist")

        X_test_sub_2_cls, y_test_sub_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="chest",
                                                                                      resp=False)

        ############  ----------------------------- Preliminary Phase -----------------------------  ############

        # RF, CNN, TRANS, LSTM - chest - hrv eda
        X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls = dataLoading.provide_test_data_fused(sub_id,
                                                                                                             option="chest",
                                                                                                             hrv=True,
                                                                                                             eda=True,
                                                                                                             resp=False)

        # RF, CNN, TRANS, LSTM - chest - hrv eda
        x_test_sub_chest_full_3cls, y_test_sub_chest_full_3cls = dataLoading.provide_test_data_fused(sub_id,
                                                                                                     option="chest",
                                                                                                     hrv=True, eda=True,
                                                                                                     resp=True)
        acc_rf_3cls_chest_full, y_pred_rf_3cls_chest_full, raw_pred_rf_3cls_chest_full = smu.predict_model(
            trained_models_chest_full['RF'], x_test_sub_chest_full_3cls, y_test_sub_chest_full_3cls, 'RF')
        acc_cnn_3cls_chest_full, y_pred_cnn_3cls_chest_full, raw_pred_cnn_3cls_chest_full = smu.predict_model(
            trained_models_chest_full['CNN'], x_test_sub_chest_full_3cls, y_test_sub_chest_full_3cls, 'CNN')
        acc_trans_3cls_chest_full, y_pred_trans_3cls_chest_full, _ = smu.predict_model(
            trained_models_chest_full['TRANS'], x_test_sub_chest_full_3cls, y_test_sub_chest_full_3cls, 'TRANS')
        acc_lstm_3cls_chest_full, y_pred_lstm_3cls_chest_full, _ = smu.predict_model(trained_models_chest_full['LSTM'],
                                                                                     x_test_sub_chest_full_3cls,
                                                                                     y_test_sub_chest_full_3cls, 'LSTM')

        ###########  --------------- CONFUSION MATRICES ---------------  ###########
        print(f"  Displaying Confusion Matrices for {sub_id}...")
        skip_plots = False
        if skip_plots:
            print("skipped plots")
        else:
            # normal models
            # chest - full
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_rf_3cls_chest_full,
                y_pred_cnn_3cls_chest_full,
                class_names,
                notation=notation_cf
            )
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_trans_3cls_chest_full,
                y_pred_lstm_3cls_chest_full,
                class_names,
                notation=notation_cf
            )

    ###########  --------------- ###################### ---------------  ###########



if __name__ == "__main__":
    main()