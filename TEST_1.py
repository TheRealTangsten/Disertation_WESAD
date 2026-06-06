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
    models_to_train = ['RF', 'CNN', 'TRANS', 'LSTM']
    trained_models_chest_hrv_eda = {}
    trained_models_chest_full = {}
    trained_models_wrist_full = {}


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

    for sub_id in TEST_SUBJECTS:
        _, y_test_sub_chest_3_cls = dataLoading.provide_test_data_fused(sub_id=sub_id, option="chest")
        _, y_test_sub_wrist_3_cls = dataLoading.provide_test_data_fused(sub_id=sub_id, option="wrist")
        _, y_test_sub_chest_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="chest")
        _, y_test_sub_wrist_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="wrist")

        X_test_sub_2_cls, y_test_sub_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="chest", resp=False)


        ###### ---------------------  2 CLASSES   --------------------- ######

        # Normal RF, Multi CNN - chest - full
        xt2_chest_full_fused, yt2_chest_full_fused = dataLoading.provide_test_data_fused_2cls(sub_id, option="chest", hrv=True, eda=True, resp=True)
        xt2_chest_full_concat, yt2_chest_full_concat, _ = dataLoading.provide_test_data_concat_2cls(sub_id, option="chest", hrv=True, eda=True, resp=True)
        acc_RF_2cls_chest_full, y_pred_RF_2cls_chest_full, _ = smu.predict_model(RF_2cls_chest_full, xt2_chest_full_fused, yt2_chest_full_fused, 'RF')
        acc_multi_CNN_2cls_chest_full, y_pred_multi_CNN_2cls_chest_full, _ = smu.predict_multi_branch_by_vector_count(multi_CNN_2cls_chest_full, xt2_chest_full_concat, yt2_chest_full_concat)

        # Normal RF, Normal CNN - chest - hrv eda
        xt2_chest_hrv_eda_fused, yt2_chest_hrv_eda_fused = dataLoading.provide_test_data_fused_2cls(sub_id, option="chest", hrv=True, eda=True, resp=False)
        acc_RF_2cls_chest_hrv_eda, y_pred_RF_2cls_chest_hrv_eda, _ = smu.predict_model(RF_2cls_chest_hrv_eda, xt2_chest_hrv_eda_fused, yt2_chest_hrv_eda_fused, 'RF')
        acc_CNN_2cls_chest_hrv_eda, y_pred_CNN_2cls_chest_hrv_eda, _ = smu.predict_model(CNN_2cls_chest_hrv_eda, xt2_chest_hrv_eda_fused, yt2_chest_hrv_eda_fused, 'CNN')

        #Multi RF, Normal LSTM - wrist
        xt2_wrist_fused, yt2_wrist_fused = dataLoading.provide_test_data_fused_2cls(sub_id, option="wrist", hrv=True, eda=True)
        xt2_wrist_concat, yt2_wrist_concat, _ = dataLoading.provide_test_data_concat_2cls(sub_id, option="wrist", hrv=True, eda=True)
        acc_multi_RF_2cls_wrist_full, y_pred_multi_RF_2cls_wrist_full, _ = smu.predict_multi_rf_independent_branches(multi_RF_2cls_wrist_full, xt2_wrist_concat, yt2_wrist_concat)
        acc_LSTM_2cls_wrist_full, y_pred_LSTM_2cls_wrist_full, _ = smu.predict_model(LSTM_2cls_wrist_full, xt2_wrist_fused, yt2_wrist_fused,'LSTM')

        ###### ---------------------  ######   --------------------- ######
        #print(f"\n  Result {sub_id}: RF={acc_rf_3cls_chest_hrv_eda:.2f}, CNN={acc_cnn_3cls_chest_hrv_eda:.2f}, Transformer={acc_trans_3cls_chest_hrv_eda:.2f}, LSTM={acc_lstm_3cls_chest_hrv_eda:.2f}")

        # RESULTS

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

        ###########  --------------- CONFUSION MATRICES ---------------  ###########

    ###########  --------------- ###################### ---------------  ###########

    # --- AFISARE TABELE PER SUBIECT ---

    # 2 Classes
    print("\n\n====== 2 CLASSES  ======")
    df_results_2cls_chest_full = pd.DataFrame(results_2cls_chest_full)
    df_results_2cls_chest_hrv_eda = pd.DataFrame(results_2cls_chest_hrv_eda)
    df_results_2cls_wrist_full = pd.DataFrame(results_2cls_wrist_full)

    print("\n=== CHEST - FULL ===")
    print(df_results_2cls_chest_full.to_string(index=False))
    if not df_results_2cls_chest_full.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"RF:  {df_results_2cls_chest_full['acc_rf'].mean():.2f}")
        print(f"CNN: {df_results_2cls_chest_full['acc_cnn'].mean():.2f}")

    print("\n=== CHEST - HRV EDA ===")
    print(df_results_2cls_chest_hrv_eda.to_string(index=False))
    if not df_results_2cls_chest_hrv_eda.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"RF:  {df_results_2cls_chest_hrv_eda['acc_rf'].mean():.2f}")
        print(f"CNN: {df_results_2cls_chest_hrv_eda['acc_cnn'].mean():.2f}")

    print("\n=== WRIST - FULL ===")
    print(df_results_2cls_wrist_full.to_string(index=False))
    if not df_results_2cls_wrist_full.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"RF:  {df_results_2cls_wrist_full['acc_rf'].mean():.2f}")
        print(f"LSTM: {df_results_2cls_wrist_full['acc_lstm'].mean():.2f}")

    # Mega Fusion

    # print(f"Multi RF: {df_results['acc_multi_rf'].mean():.2f}")
    # print(f"RF 2 classes: {df_results['acc_rf_2_cls'].mean():.2f}")




if __name__ == "__main__":
    main()