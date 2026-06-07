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
    models_to_train = ['RF', 'CNN', 'TRANS', 'LSTM']
    trained_models_chest_hrv_eda = {}
    trained_models_chest_full = {}
    trained_models_wrist_full = {}



    #Classic Train: RF, CNN, TRANS, LSTM
    for m_name in models_to_train:
        #trained_models_chest_hrv_eda[m_name] = smu.train_model(X_train_chest_hrv_eda, y_train, num_classes_3, m_name)
        trained_models_chest_hrv_eda[m_name] = smu.train_model(x3_train_chest_hrv_eda, y3_train_chest_hrv_eda, num_classes_3, m_name)
        trained_models_chest_full[m_name] = smu.train_model(x3_train_chest_full, y3_train_chest_full, num_classes_3, m_name)
        trained_models_wrist_full[m_name] = smu.train_model(x3_train_wrist_full, y3_train_wrist_full, num_classes_3, m_name)

    # Multi CNN, LSTM, RF - chest - hrv eda
    xt3_chest_hrv_eda, yt3_chest_hrv_eda, _ = dataLoading.provide_train_data_concat(option = "chest", hrv = True, eda = True, resp = False)
    multi_CNN_3cls_chest_hrv_eda = smu.train_multi_branch_by_vector_count(xt3_chest_hrv_eda, yt3_chest_hrv_eda, num_classes_3)
    multi_LSTM_3cls_chest_hrv_eda = smu.train_multi_branch_lstm_by_vector_count(xt3_chest_hrv_eda, yt3_chest_hrv_eda, num_classes_3)
    multi_RF_3cls_chest_hrv_eda = smu.train_multi_rf_independent_branches(xt3_chest_hrv_eda, yt3_chest_hrv_eda, num_classes_3)

    #Multi CNN, LSTM, RF, - chest - full
    xt3_chest_full, yt3_chest_full, _ = dataLoading.provide_train_data_concat(option="chest", hrv=True, eda=True, resp=True)
    multi_CNN_3cls_chest_full = smu.train_multi_branch_by_vector_count(xt3_chest_full, yt3_chest_full, num_classes_3)
    multi_LSTM_3cls_chest_full = smu.train_multi_branch_lstm_by_vector_count(xt3_chest_full, yt3_chest_full, num_classes_3)
    multi_RF_3cls_chest_full = smu.train_multi_rf_independent_branches(xt3_chest_full, yt3_chest_full, num_classes_3)

    #Mutli CNN, LSTM, RF - wrist - full
    xt3_wrist_full, yt3_wrist_full, _ = dataLoading.provide_train_data_concat(option = "wrist", hrv = True, eda = True)
    multi_CNN_3cls_wrist_full = smu.train_multi_branch_by_vector_count(xt3_wrist_full, yt3_wrist_full, num_classes_3)
    multi_LSTM_3cls_wrist_full = smu.train_multi_branch_lstm_by_vector_count(xt3_wrist_full, yt3_wrist_full, num_classes_3)
    multi_RF_3cls_wrist_full = smu.train_multi_rf_independent_branches(xt3_wrist_full, yt3_wrist_full, num_classes_3)

    # onesig CNN - 3 cls - chest
    onesig_CNN_chest_hrv = smu.train_model(x3_train_chest_hrv, y3_train_chest_hrv, num_classes_3, 'CNN')
    onesig_CNN_chest_eda = smu.train_model(x3_train_chest_eda, y3_train_chest_eda, num_classes_3, 'CNN')
    onesig_CNN_chest_resp = smu.train_model(x3_train_chest_resp, y3_train_chest_resp, num_classes_3, 'CNN')

    # onesig LSTM - 3 cls - chest
    onesig_LSTM_chest_hrv = smu.train_model(x3_train_chest_hrv, y3_train_chest_hrv, num_classes_3, 'LSTM')
    onesig_LSTM_chest_eda = smu.train_model(x3_train_chest_eda, y3_train_chest_eda, num_classes_3, 'LSTM')
    onesig_LSTM_chest_resp = smu.train_model(x3_train_chest_resp, y3_train_chest_resp, num_classes_3, 'LSTM')

    # onesig CNN - 3 cls - wrist
    onesig_CNN_wrist_hrv = smu.train_model(x3_train_wrist_hrv, y3_train_wrist_hrv, num_classes_3, 'CNN')
    onesig_CNN_wrist_eda = smu.train_model(x3_train_wrist_eda, y3_train_wrist_eda, num_classes_3, 'CNN')

    # onesig LSTM - 3 cls - wrist
    onesig_LSTM_wrist_hrv = smu.train_model(x3_train_wrist_hrv, y3_train_wrist_hrv, num_classes_3, 'LSTM')
    onesig_LSTM_wrist_eda = smu.train_model(x3_train_wrist_eda, y3_train_wrist_eda, num_classes_3, 'LSTM')

    ###### ---------------------  2 CLASSES   --------------------- ######
    # CNN, LSTM, RF - 2 cls - chest - hrv,eda
    x2_chest_hrv_eda, y2_chest_hrv_eda, num_cls_binary =  dataLoading.provide_train_data_fused_2cls(option="chest", hrv = True, eda = True, resp=False)
    CNN_2cls_chest_hrv_eda = smu.train_model(x2_chest_hrv_eda, y2_chest_hrv_eda, num_classes_2, 'CNN')
    #LSTM_2cls_chest_hrv_eda = smu.train_model(x2_chest_hrv_eda, y2_chest_hrv_eda, num_classes_2, 'LSTM')
    RF_2cls_chest_hrv_eda = smu.train_model(x2_chest_hrv_eda, y2_chest_hrv_eda, num_classes_2, 'RF')

    # CNN, LSTM, RF - 2 cls - chest - full
    x2_chest_full, y2_chest_full, num_cls_binary = dataLoading.provide_train_data_fused_2cls(option="chest", hrv = True, eda = True, resp=True)
    xx2_chest_full, yy2_chest_full, num_cls_binary = dataLoading.provide_train_data_concat_2cls(option="chest", hrv = True, eda = True, resp=True)
    #CNN_2cls_chest_full = smu.train_model(x2_chest_full, y2_chest_full, num_classes_2, 'CNN')
    #LSTM_2cls_chest_full = smu.train_model(x2_chest_full, y2_chest_full, num_classes_2, 'LSTM')
    multi_CNN_2cls_chest_full = smu.train_multi_branch_by_vector_count(xx2_chest_full, yy2_chest_full, num_classes_2)
    RF_2cls_chest_full = smu.train_model(x2_chest_full, y2_chest_full, num_classes_2, 'RF')

    # CNN, LSTM, RF - 2 cls - wrist - full
    x2_wrist_full, y2_wrist_full, num_cls_binary = dataLoading.provide_train_data_fused_2cls(option="wrist", hrv = True, eda = True)
    xx2_wrist_full, yy2_wrist_full, num_cls_binary = dataLoading.provide_train_data_concat_2cls(option="wrist", hrv = True, eda = True)
    #CNN_2cls_wrist_full = smu.train_model(x2_wrist_full, y2_wrist_full, num_classes_2, 'CNN')
    #RF_2cls_wrist_full = smu.train_model(x2_wrist_full, y2_wrist_full, num_classes_2, 'RF')
    LSTM_2cls_wrist_full = smu.train_model(x2_wrist_full, y2_wrist_full, num_classes_2, 'LSTM')
    multi_RF_2cls_wrist_full = smu.train_multi_rf_independent_branches(xx2_wrist_full, yy2_wrist_full, num_classes_2)


    ###### ---------------------  ######   --------------------- ######


    # --- EVALUARE PE FIECARE SUBIECT DE TEST ---
    print("\n=== STARTING EVALUATION ON TEST SUBJECTS ===")
    results_3cls_chest_hrv_eda = []
    results_3cls_chest_full = []
    results_3cls_wrist_full =[]

    results_model_fusion_chest_full = []
    results_model_fusion_chest_hrv_eda = []
    results_model_fusion_wrist = []

    results_decision_fusion = []
    results_decision_fusion_2 = []

    results_2cls_chest_hrv_eda = []
    results_2cls_chest_full = []
    results_2cls_wrist_full =[]

    results_fdf_chest_full = []
    results_fdf_chest_hrv_eda = []
    results_fdf_wrist = []

    for sub_id in TEST_SUBJECTS:
        _, y_test_sub_chest_3_cls = dataLoading.provide_test_data_fused(sub_id=sub_id, option="chest")
        _, y_test_sub_wrist_3_cls = dataLoading.provide_test_data_fused(sub_id=sub_id, option="wrist")
        _, y_test_sub_chest_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="chest")
        _, y_test_sub_wrist_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="wrist")

        X_test_sub_2_cls, y_test_sub_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="chest", resp=False)

        ############  ----------------------------- Preliminary Phase -----------------------------  ############

        # RF, CNN, TRANS, LSTM - chest - hrv eda
        X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls = dataLoading.provide_test_data_fused(sub_id, option="chest", hrv=True, eda=True, resp=False)
        acc_rf_3cls_chest_hrv_eda, y_pred_rf_3cls_chest_hrv_eda, raw_pred_rf_3cls_chest_hrv_eda = smu.predict_model(trained_models_chest_hrv_eda['RF'], X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls, 'RF')
        acc_cnn_3cls_chest_hrv_eda, y_pred_cnn_3cls_chest_hrv_eda, raw_pred_cnn_3cls_chest_hrv_eda = smu.predict_model(trained_models_chest_hrv_eda['CNN'], X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls, 'CNN')
        acc_trans_3cls_chest_hrv_eda, y_pred_trans_3cls_chest_hrv_eda, _ = smu.predict_model(trained_models_chest_hrv_eda['TRANS'], X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls, 'TRANS')
        acc_lstm_3cls_chest_hrv_eda, y_pred_lstm_3cls_chest_hrv_eda, _ = smu.predict_model(trained_models_chest_hrv_eda['LSTM'], X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls, 'LSTM')

        # RF, CNN, TRANS, LSTM - chest - full
        x_test_sub_chest_full_3cls, y_test_sub_chest_full_3cls = dataLoading.provide_test_data_fused(sub_id, option="chest", hrv=True, eda=True, resp=True)
        acc_rf_3cls_chest_full, y_pred_rf_3cls_chest_full, raw_pred_rf_3cls_chest_full = smu.predict_model(trained_models_chest_full['RF'],x_test_sub_chest_full_3cls, y_test_sub_chest_full_3cls, 'RF' )
        acc_cnn_3cls_chest_full, y_pred_cnn_3cls_chest_full, raw_pred_cnn_3cls_chest_full = smu.predict_model(trained_models_chest_full['CNN'],x_test_sub_chest_full_3cls, y_test_sub_chest_full_3cls, 'CNN')
        acc_trans_3cls_chest_full, y_pred_trans_3cls_chest_full, _ = smu.predict_model(trained_models_chest_full['TRANS'],x_test_sub_chest_full_3cls, y_test_sub_chest_full_3cls, 'TRANS')
        acc_lstm_3cls_chest_full, y_pred_lstm_3cls_chest_full, _ = smu.predict_model(trained_models_chest_full['LSTM'],x_test_sub_chest_full_3cls, y_test_sub_chest_full_3cls, 'LSTM')

        # RF, CNN, TRANS, LSTM - wrist - full
        x_test_sub_wrist_full_3cls, y_test_sub_wrist_full_3cls = dataLoading.provide_test_data_fused(sub_id, option="wrist", hrv=True, eda=True)
        acc_rf_3cls_wrist_full, y_pred_rf_3cls_wrist_full, raw_pred_rf_3cls_wrist_full = smu.predict_model(trained_models_wrist_full['RF'], x_test_sub_wrist_full_3cls, y_test_sub_wrist_full_3cls, 'RF')
        acc_cnn_3cls_wrist_full, y_pred_cnn_3cls_wrist_full, _ = smu.predict_model(trained_models_wrist_full['CNN'], x_test_sub_wrist_full_3cls, y_test_sub_wrist_full_3cls, 'CNN')
        acc_trans_3cls_wrist_full, y_pred_trans_3cls_wrist_full, _ = smu.predict_model(trained_models_wrist_full['TRANS'], x_test_sub_wrist_full_3cls, y_test_sub_wrist_full_3cls, 'TRANS')
        acc_lstm_3cls_wrist_full, y_pred_lstm_3cls_wrist_full, raw_pred_LSTM_wrist_full = smu.predict_model(trained_models_wrist_full['LSTM'], x_test_sub_wrist_full_3cls, y_test_sub_wrist_full_3cls, 'LSTM')

        ############  ----------------------------- ################### -----------------------------  ############

        acc_rf_2_cls_hrv_eda, y_pred_rf_2_cls_hrv_eda, _ = smu.predict_model(RF_2cls_chest_hrv_eda, X_test_sub_2_cls, y_test_sub_2_cls, 'RF')



        ## CNN, LSTM specializat RESPIBAN - Full - 3 cls
        xxt3_chest_full, yyt3_chest_full, _ = dataLoading.provide_test_data_concat(sub_id, option="chest", hrv=True,eda=True, resp=True)
        acc_multi_cnn_chest_3cls_full, y_pred_multi_cnn_chest_3cls_full, raw_pred_multi_cnn_3cls_chest_full = smu.predict_multi_branch_by_vector_count(
            multi_CNN_3cls_chest_full, xxt3_chest_full, yyt3_chest_full)
        acc_multi_lstm_chest_3cls_full, y_pred_multi_lstm_chest_3cls_full, _ = smu.predict_multi_branch_lstm_by_vector_count(
            multi_LSTM_3cls_chest_full, xxt3_chest_full, yyt3_chest_full)
        acc_multi_rf_chest_3cls_full, y_pred_multi_rf_chest_3cls_full, _ = smu.predict_multi_rf_independent_branches(
            multi_RF_3cls_chest_full, xxt3_chest_full, yyt3_chest_full)


        #CNN, LSTM specializat RESPIBAN - hrv/eda - 3 cls
        xxt3_chest_hrv_eda, yyt3_chest_hrv_eda, _ = dataLoading.provide_test_data_concat(sub_id, option="chest",hrv=True, eda=True, resp=False)
        acc_multi_cnn_chest_3cls_hrv_eda, y_pred_multi_cnn_chest_3cls_hrv_eda, raw_pred_multi_cnn_3cls_chest_hrv_eda = smu.predict_multi_branch_by_vector_count(
            multi_CNN_3cls_chest_hrv_eda, xxt3_chest_hrv_eda, yyt3_chest_hrv_eda)
        acc_multi_lstm_chest_3cls_hrv_eda, y_pred_multi_lstm_chest_3cls_hrv_eda, raw_pred_lstm_multi_chest_3cls_hrv_eda = smu.predict_multi_branch_lstm_by_vector_count(
            multi_LSTM_3cls_chest_hrv_eda, xxt3_chest_hrv_eda, yyt3_chest_hrv_eda)
        # RF specializat RESPIBAN - hrv/eda - 3 cls
        acc_multi_rf_chest_3cls_hrv_eda, y_pred_multi_rf_chest_3cls_hrv_eda, _ = smu.predict_multi_rf_independent_branches(
            multi_RF_3cls_chest_hrv_eda, xxt3_chest_hrv_eda, yyt3_chest_hrv_eda)

        #CNN, LSTM specializat Empatica E4 - Full - 3 cls
        xxt3_wrist_full, yyt3_wrist_full, _ = dataLoading.provide_test_data_concat(sub_id, option="wrist", hrv=True,eda=True, resp=False)
        acc_multi_cnn_wrist_3cls_full, y_pred_multi_cnn_wrist_3cls_full, _ = smu.predict_multi_branch_by_vector_count(multi_CNN_3cls_wrist_full,xxt3_wrist_full,yyt3_wrist_full)
        acc_multi_lstm_wrist_3cls_full, y_pred_multi_lstm_wrist_3cls_full, _ = smu.predict_multi_branch_lstm_by_vector_count(multi_LSTM_3cls_wrist_full, xxt3_wrist_full, yyt3_wrist_full)

        # RF specializat Empatica E4 - Full - 3 cls
        acc_multi_rf_3cls_wrist, y_pred_multi_rf_3cls_wrist_full, raw_pred_multi_rf_3cls_wrist_full = smu.predict_multi_rf_independent_branches(multi_RF_3cls_wrist_full, xxt3_wrist_full, yyt3_wrist_full)

        #onesig CNN - chest - full + hrv/eda
        xxt3_chest_hrv, yyt3_chest_hrv = dataLoading.provide_test_data_fused(sub_id, option="chest", hrv=True, eda =False, resp=False)
        xxt3_chest_eda, yyt3_chest_eda = dataLoading.provide_test_data_fused(sub_id, option="chest", hrv=False, eda=True, resp=False)
        xxt3_chest_resp, yyt3_chest_resp = dataLoading.provide_test_data_fused(sub_id, option="chest", hrv=False, eda=False, resp=True)
        acc_onesig_CNN_chest_hrv, y_pred_onesig_CNN_chest_hrv, raw_pred_CNN_chest_hrv = smu.predict_model(onesig_CNN_chest_hrv, xxt3_chest_hrv, yyt3_chest_hrv, 'CNN')
        acc_onesig_CNN_chest_eda, y_pred_onesig_CNN_chest_eda, raw_pred_CNN_chest_eda = smu.predict_model(onesig_CNN_chest_eda, xxt3_chest_eda, yyt3_chest_eda, 'CNN')
        acc_onesig_CNN_chest_resp, y_pred_onesig_CNN_chest_resp, raw_pred_CNN_chest_resp = smu.predict_model(onesig_CNN_chest_resp, xxt3_chest_resp, yyt3_chest_resp, 'CNN')

        list_raw_preds_onesig_CNN_chest_full = [raw_pred_CNN_chest_hrv, raw_pred_CNN_chest_eda, raw_pred_CNN_chest_resp]
        list_raw_preds_onesig_CNN_chest_hrv_eda = [raw_pred_CNN_chest_hrv, raw_pred_CNN_chest_eda]
        acc_onesig_CNN_chest_full, y_pred_onesig_CNN_chest_full, _ = smu.combine_results_multiple_models(list_raw_preds_onesig_CNN_chest_full, yyt3_chest_hrv)
        acc_onesig_CNN_chest_hrv_eda, y_pred_onesig_CNN_chest_hrv_eda, _ = smu.combine_results_multiple_models(list_raw_preds_onesig_CNN_chest_hrv_eda, yyt3_chest_hrv)

        #onesig LSTM - chest - full + hrv/eda
        acc_onesig_LSTM_chest_hrv, y_pred_onesig_LSTM_chest_hrv, raw_pred_LSTM_chest_hrv = smu.predict_model(onesig_LSTM_chest_hrv, xxt3_chest_hrv, yyt3_chest_hrv, 'LSTM')
        acc_onesig_LSTM_chest_eda, y_pred_onesig_LSTM_chest_eda, raw_pred_LSTM_chest_eda = smu.predict_model(onesig_LSTM_chest_eda, xxt3_chest_eda, yyt3_chest_eda, 'LSTM')
        acc_onesig_LSTM_chest_resp, y_pred_onesig_LSTM_chest_resp, raw_pred_LSTM_chest_resp = smu.predict_model(onesig_LSTM_chest_resp, xxt3_chest_resp, yyt3_chest_resp, 'LSTM')

        list_raw_preds_onesig_LSTM_chest_full = [raw_pred_LSTM_chest_hrv, raw_pred_LSTM_chest_eda, raw_pred_LSTM_chest_resp]
        list_raw_preds_onesig_LSTM_chest_eda_hrv = [raw_pred_LSTM_chest_hrv, raw_pred_LSTM_chest_eda]
        acc_onesig_LSTM_chest_full, y_pred_onesig_LSTM_chest_full, _ = smu.combine_results_multiple_models(list_raw_preds_onesig_LSTM_chest_full, yyt3_chest_hrv)
        acc_onesig_LSTM_chest_eda_hrv, y_pred_onesig_LSTM_chest_hrv_eda, _ = smu.combine_results_multiple_models(list_raw_preds_onesig_LSTM_chest_eda_hrv, yyt3_chest_hrv)

        #onesig CNN - wrist
        xxt3_wrist_hrv, yyt3_wrist_hrv = dataLoading.provide_test_data_fused(sub_id, option="wrist", hrv=True, eda=False)
        xxt3_wrist_eda, yyt3_wrist_eda = dataLoading.provide_test_data_fused(sub_id, option="wrist", hrv=False, eda=True)
        acc_onesig_CNN_wrist_hrv, y_pred_onesig_CNN_wrist_hrv, raw_pred_CNN_wrist_hrv = smu.predict_model(onesig_CNN_wrist_hrv, xxt3_wrist_hrv, yyt3_wrist_hrv, 'CNN')
        acc_onesig_CNN_wrist_eda, y_pred_onesig_CNN_wrist_eda, raw_pred_CNN_wrist_eda = smu.predict_model(onesig_CNN_wrist_eda, xxt3_wrist_eda, yyt3_wrist_eda, 'CNN')

        list_raw_preds_onesig_CNN_wrist = [raw_pred_CNN_wrist_hrv, raw_pred_CNN_wrist_eda]
        acc_onesig_CNN_wrist, y_pred_onesig_CNN_wrist, _ = smu.combine_results_multiple_models(list_raw_preds_onesig_CNN_wrist, yyt3_wrist_hrv)

        #onesig LSTM - wrist
        acc_onesig_LSTM_wrist_hrv, y_pred_onesig_LSTM_wrist_hrv, raw_pred_LSTM_wrist_hrv = smu.predict_model(onesig_LSTM_wrist_hrv, xxt3_wrist_hrv, yyt3_wrist_hrv, 'LSTM')
        acc_onesig_LSTM_wrist_eda, y_pred_onesig_LSTM_wrist_eda, raw_pred_LSTM_wrist_eda = smu.predict_model(onesig_LSTM_wrist_eda, xxt3_wrist_eda, yyt3_wrist_eda, 'LSTM')

        list_raw_preds_onesig_LSTM_wrist = [raw_pred_LSTM_wrist_hrv, raw_pred_LSTM_wrist_eda]
        acc_onesig_LSTM_wrist, y_pred_onesig_LSTM_wrist, raw_preds_onesig_LSTM_wrist = smu.combine_results_multiple_models(list_raw_preds_onesig_LSTM_wrist, yyt3_wrist_hrv)

        ###### ---------------------  2 CLASSES   --------------------- ######

        # Normal RF, Multi CNN - chest - full
        xt2_chest_full_fused, yt2_chest_full_fused = dataLoading.provide_test_data_fused_2cls(sub_id, option="chest", hrv=True, eda=True, resp=True)
        xt2_chest_full_concat, yt2_chest_full_concat, _ = dataLoading.provide_test_data_concat_2cls(sub_id, option="chest", hrv=True, eda=True, resp=True)
        acc_RF_2cls_chest_full, y_pred_RF_2cls_chest_full, raw_pred_rf_2cls_chest_full = smu.predict_model(RF_2cls_chest_full, xt2_chest_full_fused, yt2_chest_full_fused, 'RF')
        acc_multi_CNN_2cls_chest_full, y_pred_multi_CNN_2cls_chest_full, raw_pred_cnn_2cls_chest_full = smu.predict_multi_branch_by_vector_count(multi_CNN_2cls_chest_full, xt2_chest_full_concat, yt2_chest_full_concat)

        # Normal RF, Normal CNN - chest - hrv eda
        xt2_chest_hrv_eda_fused, yt2_chest_hrv_eda_fused = dataLoading.provide_test_data_fused_2cls(sub_id, option="chest", hrv=True, eda=True, resp=False)
        acc_RF_2cls_chest_hrv_eda, y_pred_RF_2cls_chest_hrv_eda, raw_pred_rf_2cls_chest_hrv_eda = smu.predict_model(RF_2cls_chest_hrv_eda, xt2_chest_hrv_eda_fused, yt2_chest_hrv_eda_fused, 'RF')
        acc_CNN_2cls_chest_hrv_eda, y_pred_CNN_2cls_chest_hrv_eda, raw_pred_cnn_2cls_chest_hrv_eda = smu.predict_model(CNN_2cls_chest_hrv_eda, xt2_chest_hrv_eda_fused, yt2_chest_hrv_eda_fused, 'CNN')

        #Multi RF, Normal LSTM - wrist
        xt2_wrist_fused, yt2_wrist_fused = dataLoading.provide_test_data_fused_2cls(sub_id, option="wrist", hrv=True, eda=True)
        xt2_wrist_concat, yt2_wrist_concat, _ = dataLoading.provide_test_data_concat_2cls(sub_id, option="wrist", hrv=True, eda=True)
        acc_multi_RF_2cls_wrist_full, y_pred_multi_RF_2cls_wrist_full, raw_pred_multi_RF_2cls_wrist_full = smu.predict_multi_rf_independent_branches(multi_RF_2cls_wrist_full, xt2_wrist_concat, yt2_wrist_concat)
        acc_LSTM_2cls_wrist_full, y_pred_LSTM_2cls_wrist_full, _ = smu.predict_model(LSTM_2cls_wrist_full, xt2_wrist_fused, yt2_wrist_fused,'LSTM')

        ###### ---------------------  ############   --------------------- ######


        ###### ---------------------  FINAL PER DATASET FUSION   --------------------- ######

        # Chest - FULL
        # Max acc: normal RF + Multi CNN
        list_raw_max_acc_chest_full = [raw_pred_rf_3cls_chest_full, raw_pred_multi_cnn_3cls_chest_full]
        acc_max_acc_chest_full, y_pred_max_acc_chest_full, _ = smu.combine_results_multiple_models(list_raw_max_acc_chest_full, y_test_sub_chest_full_3cls)
        # Class combo: normal RF + 2 cls RF
        list_raw_class_combo_chest_full = [raw_pred_rf_3cls_chest_full, raw_pred_rf_2cls_chest_full]
        acc_class_combo_chest_full, y_pred_class_combo_chest_full, _ = smu.combine_results_single_3cls_plus_2cls(raw_pred_rf_3cls_chest_full, raw_pred_rf_2cls_chest_full, y_test_sub_chest_full_3cls)
        # DL var: normal CNN + multi CNN
        list_raw_dl_var_chest_full = [raw_pred_cnn_3cls_chest_full, raw_pred_multi_cnn_3cls_chest_full]
        # acc_dl_var_chest_full, y_pred_dl_var_chest_full, _ = smu.combine_results_multiple_models(list_raw_dl_var_chest_full, y_test_sub_chest_full_3cls)
        acc_dl_var_chest_full, y_pred_dl_var_chest_full, _ = smu.combine_results_single_3cls_plus_2cls(raw_pred_cnn_3cls_chest_full, raw_pred_cnn_2cls_chest_full, y_test_sub_chest_full_3cls)

        # Chest - hrv/eda
        # Max acc: normal RF + Multi CNN
        list_raw_max_acc_chest_hrv_eda = [raw_pred_rf_3cls_chest_hrv_eda, raw_pred_multi_cnn_3cls_chest_hrv_eda]
        acc_max_acc_chest_hrv_eda, y_pred_max_acc_chest_hrv_eda, _ = smu.combine_results_multiple_models(list_raw_max_acc_chest_hrv_eda, y_test_sub_chest_hrv_eda_3_cls)
        # Class combo: normal RF + 2 cls RF
        list_raw_class_combo_chest_hrv_eda = [raw_pred_rf_3cls_chest_hrv_eda, raw_pred_rf_2cls_chest_hrv_eda]
        acc_class_combo_chest_hrv_eda, y_pred_class_combo_chest_hrv_eda, _ = smu.combine_results_single_3cls_plus_2cls(
            raw_pred_rf_3cls_chest_hrv_eda, raw_pred_rf_2cls_chest_hrv_eda, y_test_sub_chest_hrv_eda_3_cls)
        # DL var: normal CNN + 2cls CNN
        list_raw_dl_var_chest_hrv_eda = []
        acc_dl_var_chest_hrv_eda, y_pred_dl_var_chest_hrv_eda, _ = smu.combine_results_single_3cls_plus_2cls(
            raw_pred_cnn_3cls_chest_hrv_eda, raw_pred_cnn_2cls_chest_hrv_eda, y_test_sub_chest_hrv_eda_3_cls)

        # Wrist - FULL
        # Max acc: normal RF + Multi CNN
        list_raw_max_acc_wrist = [raw_pred_LSTM_wrist_full, raw_pred_multi_rf_3cls_wrist_full]
        acc_max_acc_wrist, y_pred_max_acc_wrist, _ = smu.combine_results_multiple_models(list_raw_max_acc_wrist,y_test_sub_wrist_3_cls)
        # Class combo: normal RF + 2 cls RF
        list_raw_class_combo_wrist = []
        acc_class_combo_wrist, y_pred_class_combo_wrist, _ = smu.combine_results_single_3cls_plus_2cls(raw_pred_multi_rf_3cls_wrist_full, raw_pred_multi_RF_2cls_wrist_full, y_test_sub_wrist_3_cls)
        # DL var: normal CNN + multi CNN
        list_raw_dl_var_wrist = [raw_pred_LSTM_wrist_full, raw_preds_onesig_LSTM_wrist]
        acc_dl_var_wrist, y_pred_dl_var_wrist, _ = smu.combine_results_multiple_models(list_raw_dl_var_wrist, y_test_sub_wrist_3_cls)


        ###### ---------------------  ############   --------------------- ######
        # RESULTS
        #NORMAL
        results_3cls_chest_hrv_eda.append({
            'subject': sub_id,
            'acc_rf': acc_rf_3cls_chest_hrv_eda,
            'acc_cnn': acc_cnn_3cls_chest_hrv_eda,
            'acc_transformer': acc_trans_3cls_chest_hrv_eda,
            'acc_lstm': acc_lstm_3cls_chest_hrv_eda,
        })
        results_3cls_chest_full.append({
            'subject': sub_id,
            'acc_rf': acc_rf_3cls_chest_full,
            'acc_cnn': acc_cnn_3cls_chest_full,
            'acc_transformer': acc_trans_3cls_chest_full,
            'acc_lstm': acc_lstm_3cls_chest_full,
        })
        results_3cls_wrist_full.append({
            'subject': sub_id,
            'acc_rf': acc_rf_3cls_wrist_full,
            'acc_cnn': acc_cnn_3cls_wrist_full,
            'acc_transformer': acc_trans_3cls_wrist_full,
            'acc_lstm': acc_lstm_3cls_wrist_full,
        })


        # MODEL FUSION
        results_model_fusion_chest_hrv_eda.append({
            'subject': sub_id,
            'acc_multi_cnn': acc_multi_cnn_chest_3cls_hrv_eda,
            'acc_multi_lstm': acc_multi_lstm_chest_3cls_hrv_eda,
            'acc_multi_rf': acc_multi_rf_chest_3cls_hrv_eda,
        })
        results_model_fusion_chest_full.append({
            'subject': sub_id,
            'acc_multi_cnn': acc_multi_cnn_chest_3cls_full,
            'acc_multi_lstm': acc_multi_lstm_chest_3cls_full,
            'acc_multi_rf': acc_multi_rf_chest_3cls_full,
            })
        results_model_fusion_wrist.append({
            'subject': sub_id,
            'acc_multi_cnn': acc_multi_cnn_wrist_3cls_full,
            'acc_multi_lstm': acc_multi_lstm_wrist_3cls_full,
            'acc_multi_rf': acc_multi_rf_3cls_wrist,
        })

        # DECISION FUSION
        results_decision_fusion.append({
            'subject': sub_id,
            'acc_CNN_chest_full': acc_onesig_CNN_chest_full,
            'acc_CNN_chest_hrv_eda': acc_onesig_CNN_chest_hrv_eda,
            'acc_LSTM_chest_full': acc_onesig_LSTM_chest_full,
            'acc_LSTM_chest_hrv_eda': acc_onesig_LSTM_chest_eda_hrv
        })
        results_decision_fusion_2.append({
            'subject': sub_id,
            'acc_CNN_wrist': acc_onesig_CNN_wrist,
            'acc_LSTM_wrist': acc_onesig_LSTM_wrist,
        })



        ###### ---------------------  2 CLASSES   --------------------- ######
        results_2cls_chest_full.append({
            'subject': sub_id,
            'acc_rf': acc_RF_2cls_chest_full,
            'acc_cnn': acc_multi_CNN_2cls_chest_full,
        })
        results_2cls_chest_hrv_eda.append({
            'subject': sub_id,
            'acc_rf': acc_RF_2cls_chest_hrv_eda,
            'acc_cnn': acc_CNN_2cls_chest_hrv_eda,
        })
        results_2cls_wrist_full.append({
            'subject': sub_id,
            'acc_rf': acc_multi_RF_2cls_wrist_full,
            'acc_lstm': acc_LSTM_2cls_wrist_full
        })
        ###### ---------------------  ######   --------------------- ######

        # Final per dataset fusion

        results_fdf_chest_full.append({
            'subject': sub_id,
            'max_acc': acc_max_acc_chest_full,
            'class_combo': acc_class_combo_chest_full,
            'dl_var': acc_dl_var_chest_full
        })

        results_fdf_chest_hrv_eda.append({
            'subject': sub_id,
            'max_acc': acc_max_acc_chest_hrv_eda,
            'class_combo': acc_class_combo_chest_hrv_eda,
            'dl_var': acc_dl_var_chest_hrv_eda
        })
        results_fdf_wrist.append({
            'subject': sub_id,
            'max_acc': acc_max_acc_wrist,
            'class_combo': acc_class_combo_wrist,
            'dl_var': acc_dl_var_wrist
        })
        ###########  --------------- CONFUSION MATRICES ---------------  ###########
        print(f"  Displaying Confusion Matrices for {sub_id}...")
        skip_plots = True
        if skip_plots:
            print("skipped plots")
        else:
            # normal models
            #chest - full
            model_names_1 = ["RF", "CNN"]
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_rf_3cls_chest_full,
                y_pred_cnn_3cls_chest_full,
                class_names,
                model_names=model_names_1,
                notation=notation_cf
            )
            model_names_2 = ["Transformer", "LSTM"]
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_trans_3cls_chest_full,
                y_pred_lstm_3cls_chest_full,
                class_names,
                model_names=model_names_2,
                notation=notation_cf
            )
            #chest - hrv eda

            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_hrv_eda_3_cls,
                y_pred_rf_3cls_chest_hrv_eda,
                y_pred_cnn_3cls_chest_hrv_eda,
                class_names,
                model_names=model_names_1,
                notation=notation_che
            )
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_hrv_eda_3_cls,
                y_pred_trans_3cls_chest_hrv_eda,
                y_pred_lstm_3cls_chest_hrv_eda,
                class_names,
                model_names=model_names_2,
                notation=notation_che
            )
            #wrist
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_wrist_3_cls,
                y_pred_rf_3cls_wrist_full,
                y_pred_cnn_3cls_wrist_full,
                class_names,
                model_names=model_names_1,
                notation=notation_w
            )
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_wrist_3_cls,
                y_pred_trans_3cls_wrist_full,
                y_pred_lstm_3cls_wrist_full,
                class_names,
                model_names=model_names_2,
                notation=notation_w
            )

            # Model fusion

            #Chest - full
            model_names = ['Multi CNN', 'Multi LSTM']
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_multi_cnn_chest_3cls_hrv_eda,
                y_pred_multi_lstm_chest_3cls_hrv_eda,
                class_names,
                model_names=model_names,
                notation=notation_cf
            )
            #Chest - hrv/eda
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_hrv_eda_3_cls,
                y_pred_multi_cnn_chest_3cls_hrv_eda,
                y_pred_multi_lstm_chest_3cls_hrv_eda,
                class_names,
                model_names=model_names,
                notation=notation_che
            )

            #Wrist - full
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_wrist_3_cls,
                y_pred_multi_cnn_wrist_3cls_full,
                y_pred_multi_lstm_wrist_3cls_full,
                class_names,
                model_names=model_names,
                notation=notation_w
            )

            # Decision Fusion

            model_names = ["DecFusion CNN", "DecFusion LSTM"]
            model_name = "Multi RF"
            #Chest - Full
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_onesig_CNN_chest_full,
                y_pred_onesig_LSTM_chest_full,
                class_names,
                model_names=model_names,
                notation=notation_cf
            )
            plting.plot_sub_conf_mat(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_multi_rf_chest_3cls_full,
                class_names,
                model_name=model_name,
                notation=notation_cf
            )
            #Chest - hrv/eda
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_hrv_eda_3_cls,
                y_pred_onesig_CNN_chest_hrv_eda,
                y_pred_onesig_LSTM_chest_hrv_eda,
                class_names,
                model_names=model_names,
                notation=notation_che
            )
            plting.plot_sub_conf_mat(
                sub_id,
                y_test_sub_chest_hrv_eda_3_cls,
                y_pred_multi_rf_chest_3cls_hrv_eda,
                class_names,
                model_name=model_name,
                notation=notation_che
            )
            #Wrist - Full
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_wrist_3_cls,
                y_pred_onesig_CNN_wrist,
                y_pred_onesig_LSTM_wrist,
                class_names,
                model_names=model_names,
                notation=notation_w
            )
            plting.plot_sub_conf_mat(
                sub_id,
                y_test_sub_wrist_3_cls,
                y_pred_multi_rf_3cls_wrist_full,
                class_names,
                model_name=model_name,
                notation=notation_w
            )

            #2 classes

            # Chest - Full
            model_names = ["RF", "Multi CNN"]
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_2_cls,
                y_pred_RF_2cls_chest_full,
                y_pred_multi_CNN_2cls_chest_full,
                class_names,
                model_names=model_names,
                notation=notation_cf
            )
            # Chest - Hrv/Eda
            model_names = ["RF", "Normal CNN"]
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_2_cls,
                y_pred_RF_2cls_chest_hrv_eda,
                y_pred_CNN_2cls_chest_hrv_eda,
                class_names,
                model_names=model_names,
                notation=notation_che
            )
            # Wrist
            model_names = ["Multi RF", "Normal LSTM"]
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_wrist_2_cls,
                y_pred_multi_RF_2cls_wrist_full,
                y_pred_LSTM_2cls_wrist_full,
                class_names,
                model_names=model_names,
                notation=notation_w
            )

            # Final Per Data Fusion

            # Chest - Full
            model_names = ["Normal RF + Multi CNN", "Normal RF + 2cls RF"]
            model_name = "normal CNN + 2cls CNN"
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_max_acc_chest_full,
                y_pred_class_combo_chest_full,
                class_names,
                model_names=model_names,
                notation=notation_cf
            )
            plting.plot_sub_conf_mat(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_dl_var_chest_full,
                class_names,
                model_name=model_name,
                notation=notation_cf
            )
            # Chest - Hrv/eda
            model_names = ["Normal RF + Multi CNN", "Normal RF + 2cls RF"]
            model_name = "normal CNN + 2cls CNN"
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_max_acc_chest_hrv_eda,
                y_pred_class_combo_chest_hrv_eda,
                class_names,
                model_names=model_names,
                notation=notation_che
            )
            plting.plot_sub_conf_mat(
                sub_id,
                y_test_sub_chest_3_cls,
                y_pred_dl_var_chest_hrv_eda,
                class_names,
                model_name=model_name,
                notation=notation_che
            )

            # Wrist
            model_names = ["Normal LSTM + Multi RF", "Multi RF + 2cls multi RF"]
            model_name = "Normal LSTM + DecFusion LSTM "
            plting.plot_subject_confusion_matrices_2col(
                sub_id,
                y_test_sub_wrist_3_cls,
                y_pred_max_acc_wrist,
                y_pred_class_combo_wrist,
                class_names,
                model_names=model_names,
                notation=notation_w
            )
            plting.plot_sub_conf_mat(
                sub_id,
                y_test_sub_wrist_3_cls,
                y_pred_dl_var_wrist,
                class_names,
                model_name=model_name,
                notation=notation_w
            )


    ###########  --------------- ###################### ---------------  ###########

    # --- AFISARE TABELE PER SUBIECT ---

    # Normal models
    print("\n====== NORMAL MODELS ======")
    df_results_chest_hrv_eda = pd.DataFrame(results_3cls_chest_hrv_eda)
    df_results_chest_full = pd.DataFrame(results_3cls_chest_full)
    df_results_wrist_full =  pd.DataFrame(results_3cls_wrist_full)

    print("\n=== CHEST - HRV EDA ===")
    print(df_results_chest_hrv_eda.to_string(index=False))
    if not df_results_chest_hrv_eda.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"RF:  {df_results_chest_hrv_eda['acc_rf'].mean():.2f}")
        print(f"CNN: {df_results_chest_hrv_eda['acc_cnn'].mean():.2f}")
        print(f"TRANS : {df_results_chest_hrv_eda['acc_transformer'].mean():.2f}")
        print(f"LSTM: {df_results_chest_hrv_eda['acc_lstm'].mean():.2f}")

    print("\n=== CHEST - FULL ===")
    print(df_results_chest_full.to_string(index=False))
    if not df_results_chest_full.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"RF:  {df_results_chest_full['acc_rf'].mean():.2f}")
        print(f"CNN: {df_results_chest_full['acc_cnn'].mean():.2f}")
        print(f"TRANS : {df_results_chest_full['acc_transformer'].mean():.2f}")
        print(f"LSTM: {df_results_chest_full['acc_lstm'].mean():.2f}")

    print("\n=== WRIST - FULL ===")
    print(df_results_wrist_full.to_string(index=False))
    if not df_results_wrist_full.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"RF:  {df_results_wrist_full['acc_rf'].mean():.2f}")
        print(f"CNN: {df_results_wrist_full['acc_cnn'].mean():.2f}")
        print(f"TRANS : {df_results_wrist_full['acc_transformer'].mean():.2f}")
        print(f"LSTM: {df_results_wrist_full['acc_lstm'].mean():.2f}")


    # Model Fusion
    print("\n\n====== MODEL FUSION ======")
    df_results_model_fusion_chest_full = pd.DataFrame(results_model_fusion_chest_full)
    df_results_model_fusion_chest_hrv_eda = pd.DataFrame(results_model_fusion_chest_hrv_eda)
    df_results_model_fusion_wrist = pd.DataFrame(results_model_fusion_wrist)

    print("\n=== CHEST - FULL ===")
    print(df_results_model_fusion_chest_full.to_string(index=False))
    if not df_results_model_fusion_chest_full.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"Multi CNN: {df_results_model_fusion_chest_full['acc_multi_cnn'].mean():.2f}")
        print(f"Multi LSTM: {df_results_model_fusion_chest_full['acc_multi_lstm'].mean():.2f}")
        print(f"Multi RF: {df_results_model_fusion_chest_full['acc_multi_rf'].mean():.2f}")

    print("\n=== CHEST - HRV EDA ===")
    print(df_results_model_fusion_chest_hrv_eda.to_string(index=False))
    if not df_results_model_fusion_chest_hrv_eda.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"Multi CNN: {df_results_model_fusion_chest_hrv_eda['acc_multi_cnn'].mean():.2f}")
        print(f"Multi LSTM: {df_results_model_fusion_chest_hrv_eda['acc_multi_lstm'].mean():.2f}")
        print(f"Multi RF: {df_results_model_fusion_chest_hrv_eda['acc_multi_rf'].mean():.2f}")

    print("\n=== WRIST - FULL ===")
    print(df_results_model_fusion_wrist.to_string(index=False))
    if not df_results_model_fusion_wrist.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"Multi CNN: {df_results_model_fusion_wrist['acc_multi_cnn'].mean():.2f}")
        print(f"Multi LSTM: {df_results_model_fusion_wrist['acc_multi_lstm'].mean():.2f}")
        print(f"Multi RF: {df_results_model_fusion_wrist['acc_multi_rf'].mean():.2f}")
    print("\n===  ===")


    # Decision Fusion
    print("\n\n====== DECISION FUSION  ======")
    df_results_decision_fusion = pd.DataFrame(results_decision_fusion)
    df_results_decision_fusion_2 = pd.DataFrame(results_decision_fusion_2)

    print(df_results_decision_fusion.to_string(index=False))
    if not df_results_decision_fusion.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"CNN chest full: {df_results_decision_fusion['acc_CNN_chest_full'].mean():.2f}")
        print(f"CNN chest hrv/eda: {df_results_decision_fusion['acc_CNN_chest_hrv_eda'].mean():.2f}")
        print(f"LSTM chest full: {df_results_decision_fusion['acc_LSTM_chest_full'].mean():.2f}")
        print(f"LSTM chest hrv/eda: {df_results_decision_fusion['acc_LSTM_chest_hrv_eda'].mean():.2f}")

    print(df_results_decision_fusion_2.to_string(index=False))
    if not df_results_decision_fusion_2.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"CNN wrist: {df_results_decision_fusion_2['acc_CNN_wrist'].mean():.2f}")
        print(f"LSTM wrist: {df_results_decision_fusion_2['acc_LSTM_wrist'].mean():.2f}")
    print("\n===  ===")

    # 2 Classes
    print("\n\n====== 2 CLASSES  ======")
    df_results_2cls_chest_full = pd.DataFrame(results_2cls_chest_full)
    df_results_2cls_chest_hrv_eda = pd.DataFrame(results_2cls_chest_hrv_eda)
    df_results_2cls_wrist_full = pd.DataFrame(results_2cls_wrist_full)

    print("\n=== CHEST - FULL ===")
    print(df_results_2cls_chest_full.to_string(index=False))
    if not df_results_2cls_chest_full.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"Normal RF 2 cls:  {df_results_2cls_chest_full['acc_rf'].mean():.2f}")
        print(f"Multi CNN 2 cls: {df_results_2cls_chest_full['acc_cnn'].mean():.2f}")

    print("\n=== CHEST - HRV EDA ===")
    print(df_results_2cls_chest_hrv_eda.to_string(index=False))
    if not df_results_2cls_chest_hrv_eda.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"Normal RF 2 cls:  {df_results_2cls_chest_hrv_eda['acc_rf'].mean():.2f}")
        print(f"Normal CNN 2 cls: {df_results_2cls_chest_hrv_eda['acc_cnn'].mean():.2f}")

    print("\n=== WRIST - FULL ===")
    print(df_results_2cls_wrist_full.to_string(index=False))
    if not df_results_2cls_wrist_full.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"Multi RF 2 cls:  {df_results_2cls_wrist_full['acc_rf'].mean():.2f}")
        print(f"Normal LSTM 2 cls: {df_results_2cls_wrist_full['acc_lstm'].mean():.2f}")

    # Final per dataset fusion
    df_results_fdf_chest_full = pd.DataFrame(results_fdf_chest_full)
    df_results_fdf_chest_hrv_eda = pd.DataFrame(results_fdf_chest_hrv_eda)
    df_results_fdf_wrist = pd.DataFrame(results_fdf_wrist)

    print("\n=== CHEST - FULL ===")
    print(df_results_fdf_chest_full.to_string(index=False))
    if not df_results_fdf_chest_full.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"Normal RF + Multi CNN:  {df_results_fdf_chest_full['max_acc'].mean():.2f}")
        print(f"Normal RF + 2cls RF : {df_results_fdf_chest_full['class_combo'].mean():.2f}")
        print(f"normal CNN + 2cls CNN: {df_results_fdf_chest_full['dl_var'].mean():.2f}")

    print("\n=== CHEST - HRV EDA ===")
    print(df_results_fdf_chest_hrv_eda.to_string(index=False))
    if not df_results_fdf_chest_hrv_eda.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"Normal RF + Multi CNN:  {df_results_fdf_chest_hrv_eda['max_acc'].mean():.2f}")
        print(f"Normal RF + 2cls RF: {df_results_fdf_chest_hrv_eda['class_combo'].mean():.2f}")
        print(f"Normal CNN + 2cls CNN: {df_results_fdf_chest_hrv_eda['dl_var'].mean():.2f}")

    print("\n=== WRIST - FULL ===")
    print(df_results_fdf_wrist.to_string(index=False))
    if not df_results_fdf_wrist.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"Normal LSTM + Multi RF:  {df_results_fdf_wrist['max_acc'].mean():.2f}")
        print(f"Multi RF + 2 cls multi RF: {df_results_fdf_wrist['class_combo'].mean():.2f}")
        print(f"Normal LSTM + DecFusion LSTM: {df_results_fdf_wrist['dl_var'].mean():.2f}")


if __name__ == "__main__":
    main()