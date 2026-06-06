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


# Funcția grafică păstrată din scriptul original
def plot_subject_confusion_matrices(subject_id, y_true, y_pred_rf, y_pred_cnn, y_pred_trans, y_pred_lstm, classes, model_names = ['Random Forest', 'CNN', 'Transformer', 'LSTM']):
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
    x3_train_chest_hrv_eda, y3_train_chest_hrv_eda, num_cls = dataLoading.provide_train_data_fused(option="chest", hrv=True, eda=True, resp=False)

    x3_train_chest_full, y3_train_chest_full, _ = dataLoading.provide_train_data_fused(option="chest", hrv=True, eda=True, resp=True)

    x3_train_chest_hrv, y3_train_chest_hrv, _ = dataLoading.provide_train_data_fused(option="chest", hrv=True, eda=False, resp=False)
    x3_train_chest_eda, y3_train_chest_eda, _ = dataLoading.provide_train_data_fused(option="chest", hrv=False, eda=True, resp=False)
    x3_train_chest_resp, y3_train_chest_resp, _ = dataLoading.provide_train_data_fused(option="chest", hrv=False, eda=False, resp=True)

    x3_train_wrist_full, y3_train_wrist_full, _ = dataLoading.provide_train_data_fused(option="wrist", hrv=True, eda=True)

    x3_train_wrist_hrv, y3_train_wrist_hrv, _ = dataLoading.provide_train_data_fused(option="wrist", hrv=True, eda=False)
    x3_train_wrist_eda, y3_train_wrist_eda, _ = dataLoading.provide_train_data_fused(option="wrist", hrv=False, eda=True)



    print(f"Training Data Size: {len(x3_train_chest_hrv_eda)} samples")
    #######################################################################################################################################




    # onesig CNN - 3 cls - chest
    onesig_CNN_chest_hrv = smu.train_model(x3_train_chest_hrv, y3_train_chest_hrv, num_classes_3, 'CNN')
    onesig_CNN_chest_eda = smu.train_model(x3_train_chest_eda, y3_train_chest_eda, num_classes_3, 'CNN')
    onesig_CNN_chest_resp = smu.train_model(x3_train_chest_resp, y3_train_chest_resp, num_classes_3, 'CNN')

    # onesig LSTM - 3 cls - chest
    #onesig_LSTM_chest_hrv = smu.train_model(x3_train_chest_hrv, y3_train_chest_hrv, num_classes_3, 'LSTM')
    #onesig_LSTM_chest_eda = smu.train_model(x3_train_chest_eda, y3_train_chest_eda, num_classes_3, 'LSTM')
    #onesig_LSTM_chest_resp = smu.train_model(x3_train_chest_resp, y3_train_chest_resp, num_classes_3, 'LSTM')

    # Multi CNN, LSTM, RF - chest - hrv eda
    xt3_chest_hrv_eda, yt3_chest_hrv_eda, _ = dataLoading.provide_train_data_concat(option = "chest", hrv = True, eda = True, resp = False)
    multi_CNN_3cls_chest_hrv_eda = smu.train_multi_branch_by_vector_count(xt3_chest_hrv_eda, yt3_chest_hrv_eda, num_classes_3)
    multi_LSTM_3cls_chest_hrv_eda = smu.train_multi_branch_lstm_by_vector_count(xt3_chest_hrv_eda, yt3_chest_hrv_eda, num_classes_3)
    multi_RF_3cls_chest_hrv_eda = smu.train_multi_rf_independent_branches(xt3_chest_hrv_eda, yt3_chest_hrv_eda, num_classes_3)

    # --- EVALUARE PE FIECARE SUBIECT DE TEST ---
    print("\n=== STARTING EVALUATION ON TEST SUBJECTS ===")
    results_3cls_chest_hrv_eda = []
    results_3cls_chest_full = []
    results_3cls_wrist_full =[]

    results_model_fusion = []
    results_decision_fusion = []

    for sub_id in TEST_SUBJECTS:
        _, y_test_sub_chest_3_cls = dataLoading.provide_test_data_fused(sub_id=sub_id, option="chest")
        _, y_test_sub_wrist_3_cls = dataLoading.provide_test_data_fused(sub_id=sub_id, option="wrist")
        _, y_test_sub_chest_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="chest")
        _, y_test_sub_wrist_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="wrist")

        #onesig CNN - chest - full + hrv/eda
        xxt3_chest_hrv, yyt3_chest_hrv = dataLoading.provide_test_data_fused(sub_id, option="chest", hrv=True, eda =False, resp=False)
        xxt3_chest_eda, yyt3_chest_eda = dataLoading.provide_test_data_fused(sub_id, option="chest", hrv=False, eda=True, resp=False)
        xxt3_chest_resp, yyt3_chest_resp = dataLoading.provide_test_data_fused(sub_id, option="chest", hrv=False, eda=False, resp=True)
        acc_onesig_CNN_chest_hrv, y_pred_onesig_CNN_chest_hrv, raw_pred_CNN_chest_hrv = smu.predict_model(onesig_CNN_chest_hrv, xxt3_chest_hrv, yyt3_chest_hrv, 'CNN')
        acc_onesig_CNN_chest_eda, y_pred_onesig_CNN_chest_eda, raw_pred_CNN_chest_eda = smu.predict_model(onesig_CNN_chest_eda, xxt3_chest_eda, yyt3_chest_eda, 'CNN')
        acc_onesig_CNN_chest_resp, y_pred_onesig_CNN_chest_resp, raw_pred_CNN_chest_resp = smu.predict_model(onesig_CNN_chest_resp, xxt3_chest_resp, yyt3_chest_resp, 'CNN')

        list_raw_preds_onesig_CNN_chest_full = [raw_pred_CNN_chest_hrv, raw_pred_CNN_chest_eda, raw_pred_CNN_chest_resp]
        list_raw_preds_onesig_CNN_chest_hrv_eda = [raw_pred_CNN_chest_hrv, raw_pred_CNN_chest_eda]
        acc_onesig_CNN_chest_full, y_pred_onesig_CNN_chest_full, _ = smu.combine_results_multiple_models(list_raw_preds_onesig_CNN_chest_full, y_test_sub_chest_3_cls)
        acc_onesig_CNN_chest_hrv_eda, y_pred_onesig_CNN_chest_hrv_eda, _ = smu.combine_results_multiple_models(list_raw_preds_onesig_CNN_chest_hrv_eda, y_test_sub_chest_3_cls)

        xxt3_chest_hrv_eda, yyt3_chest_hrv_eda, _ = dataLoading.provide_test_data_concat(sub_id, option="chest",hrv=True, eda=True, resp=False)
        acc_multi_cnn_chest_3cls_hrv_eda, y_pred_multi_cnn_3_chest_3cls_hrv_eda, raw_pred_multi_cnn_3_chest_3cls_hrv_eda = smu.predict_multi_branch_by_vector_count(
            multi_CNN_3cls_chest_hrv_eda, xxt3_chest_hrv_eda, yyt3_chest_hrv_eda)
        acc_multi_lstm_chest_3cls_hrv_eda, y_pred_lstm_multi_chest_3cls_hrv_eda, raw_pred_lstm_multi_chest_3cls_hrv_eda = smu.predict_multi_branch_lstm_by_vector_count\
            (multi_LSTM_3cls_chest_hrv_eda, xxt3_chest_hrv_eda, yyt3_chest_hrv_eda)

        # RESULTS
        results_3cls_chest_hrv_eda.append({
            'subject': sub_id,
            'acc_onesig_CNN_chest_full': acc_onesig_CNN_chest_full,
            'acc_onesig_CNN_chest_hrv_eda': acc_onesig_CNN_chest_hrv_eda,
            #'acc_transformer': acc_onesig_CNN_chest_full,
            #'acc_lstm': acc_onesig_CNN_chest_full,
        })


        ###########  --------------- CONFUSION MATRICES ---------------  ###########
        print(f"  Displaying Confusion Matrices for {sub_id}...")
        #class_names = ["onesig CNN chest full","onesig CNN chest full","onesig CNN chest hrv/eda","onesig CNN chest hrv/eda"]
        model_names = ["onesig CNN chest full", "onesig CNN chest full", "onesig CNN chest hrv/eda","onesig CNN chest hrv/eda"]
        skip_plots = False
        if skip_plots:
            print("skipped plots")
        else:
            # normal models, chest - hrv eda
            plot_subject_confusion_matrices(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_onesig_CNN_chest_full,
                y_pred_onesig_CNN_chest_full,
                y_pred_onesig_CNN_chest_hrv_eda,
                y_pred_onesig_CNN_chest_hrv_eda,
                class_names,
                model_names
            )

    ###########  --------------- ###################### ---------------  ###########

    # --- AFISARE TABELE PER SUBIECT ---
    print("\n====== NORMAL MODELS ======")
    # Normal models
    df_results_chest_hrv_eda = pd.DataFrame(results_3cls_chest_hrv_eda)

    print("\n=== CHEST - HRV EDA ===")
    print(df_results_chest_hrv_eda.to_string(index=False))
    if not df_results_chest_hrv_eda.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"onesig CNN chest full:  {df_results_chest_hrv_eda['acc_onesig_CNN_chest_full'].mean():.2f}")
        print(f"onesig CNN chest hrv/eda: {df_results_chest_hrv_eda['acc_onesig_CNN_chest_hrv_eda'].mean():.2f}")
        #print(f"onesig CNN chest hrv/eda : {df_results_chest_hrv_eda['acc_transformer'].mean():.2f}")
        #print(f"onesig CNN chest hrv/eda: {df_results_chest_hrv_eda['acc_lstm'].mean():.2f}")

    list_raw_preds = [raw_pred_multi_cnn_3_chest_3cls_hrv_eda, raw_pred_lstm_multi_chest_3cls_hrv_eda]
    acc_res, preds_res, raw_res = smu.combine_results_multiple_models(list_raw_preds, yyt3_chest_hrv_eda)
    print(f"ACC multi_cnn_3cls_chest_hrv_eda + multi_lstm_3cls_chest_hrv_eda: {acc_res:.2f}")


if __name__ == "__main__":
    main()