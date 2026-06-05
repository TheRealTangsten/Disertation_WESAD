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
    #######################################################################################################################################
    df_chest_hrv_eda = dataLoading.load_processed_data(json_type="chest", include_resp=False)
    full_df_wrist = dataLoading.load_processed_data(json_type="wrist", include_resp=False)
    full_df_binary = dataLoading.load_processed_data_binary(json_type="chest", include_resp=False)

    # Encodăm etichetele text în valori numerice (0, 1, 2)
    le = LabelEncoder()
    df_chest_hrv_eda['Label'] = le.fit_transform(df_chest_hrv_eda['Label'])
    full_df_wrist['Label'] = le.fit_transform(full_df_wrist['Label'])

    le2 = LabelEncoder()
    full_df_binary['Label'] = le2.fit_transform(full_df_binary['Label'])

    num_classes_3 = len(le.classes_)
    num_classes_2 = len(le2.classes_)

    # SPLIT DATE: SUBIECȚI ANTRENARE vs SUBIECȚI TEST
    print(f"\n[INFO] Splitting Data. Test Subjects: {TEST_SUBJECTS}")
    test_data_chest_hrv_eda = df_chest_hrv_eda[df_chest_hrv_eda['Subject'].isin(TEST_SUBJECTS)].copy()
    train_data_chest_hrv_eda = df_chest_hrv_eda[~df_chest_hrv_eda['Subject'].isin(TEST_SUBJECTS)].copy()

    test_data_all_2_cls = full_df_binary[full_df_binary['Subject'].isin(TEST_SUBJECTS)].copy()
    train_data_all_2_cls =  full_df_binary[~full_df_binary['Subject'].isin(TEST_SUBJECTS)].copy()

    test_data_all_wrist = full_df_wrist[full_df_wrist['Subject'].isin(TEST_SUBJECTS)].copy()
    #train_data_all_wrist = full_df_wrist[~full_df_wrist['Subject'].isin(TEST_SUBJECTS)].copy()

    # Separăm feature-urile de label/subiect pentru setul de train
    X_train_chest_hrv_eda = train_data_chest_hrv_eda.drop(columns=["Label", "Subject"])
    X_train_2_cls = train_data_all_2_cls.drop(columns=["Label", "Subject"])

    y_train = train_data_chest_hrv_eda["Label"].values
    y_train_2_cls = train_data_all_2_cls["Label"].values


    print(f"Training Data Size: {len(X_train_chest_hrv_eda)} samples")
    #######################################################################################################################################
    models_to_train = ['RF', 'CNN', 'TRANS', 'LSTM']
    trained_models_chest_hrv_eda = {}
    trained_models_chest_full = {}


    X_train_model, Y_test_model, num_cls = dataLoading.provide_train_data_fused(option = "chest", hrv = True, eda = True, resp = False)
    #Classic Train: RF, CNN, TRANS, LSTM
    for m_name in models_to_train:
        #trained_models_chest_hrv_eda[m_name] = smu.train_model(X_train_chest_hrv_eda, y_train, num_classes_3, m_name)
        trained_models_chest_hrv_eda[m_name] = smu.train_model(X_train_model, Y_test_model, num_cls, m_name)

    X_train_binary, Y_train_binary, num_cls_binary =  dataLoading.provide_train_data_fused_2cls(option="chest", resp=False)
    #RF_2_cls = smu.train_model(X_train_2_cls, y_train_2_cls, num_classes_2, 'RF')
    RF_2_cls = smu.train_model(X_train_binary, Y_train_binary, num_cls_binary, 'RF')


    xt3_chest_hrv_eda, yt3_chest_hrv_eda, _ = dataLoading.provide_train_data_concat(option = "chest", hrv = True, eda = True, resp = False)
    trained_multi_cnn_chest_3cls_hrv_eda = smu.train_multi_branch_by_vector_count(xt3_chest_hrv_eda, yt3_chest_hrv_eda, num_classes_3)
    trained_multi_lstm_chest_3cls_hrv_eda = smu.train_multi_branch_lstm_by_vector_count(xt3_chest_hrv_eda, yt3_chest_hrv_eda, num_classes_3)
    skip_some_models = False
    if skip_some_models:
        print(" SKIPPED SOME MODELS - TRAIN")
    else:
        trained_multi_rf_chest_3cls_hrv_eda = smu.train_multi_rf_independent_branches(xt3_chest_hrv_eda, yt3_chest_hrv_eda, num_classes_3)

        xt3_chest_full, yt3_chest_full, _ = dataLoading.provide_train_data_concat(option="chest", hrv=True, eda=True, resp=True)
        trained_multi_cnn_chest_3cls_full = smu.train_multi_branch_by_vector_count(xt3_chest_full, yt3_chest_full, num_classes_3)
        trained_multi_lstm_chest_3cls_full = smu.train_multi_branch_lstm_by_vector_count(xt3_chest_full, yt3_chest_full, num_classes_3)
        trained_multi_rf_chest_3cls_full = smu.train_multi_rf_independent_branches(xt3_chest_full, yt3_chest_full, num_classes_3)


        xt3_wrist_full, yt3_wrist_full, _ = dataLoading.provide_train_data_concat(option = "wrist", hrv = True, eda = True, resp = False)
        trained_multi_cnn_wrist_3cls_full = smu.train_multi_branch_by_vector_count(xt3_wrist_full, yt3_wrist_full,
                                                                                   num_classes_3)
        trained_multi_lstm_wrist_3cls_full = smu.train_multi_branch_lstm_by_vector_count(xt3_wrist_full, yt3_wrist_full,
                                                                                         num_classes_3)
        trained_multi_rf_wrist_3cls_full = smu.train_multi_rf_independent_branches(xt3_wrist_full, yt3_wrist_full,
                                                                                   num_classes_3)

        wxt, wyt, _ = dataLoading.provide_train_data_concat(option = "wrist", hrv = True, eda = True, resp = False)
        trained_multi_rf_2 = smu.train_multi_rf_independent_branches(wxt, wyt, num_classes_3)

    # --- EVALUARE PE FIECARE SUBIECT DE TEST ---
    print("\n=== STARTING EVALUATION ON TEST SUBJECTS ===")
    results = []
    results_model_fusion = []
    results_decision_fusion = []

    for sub_id in TEST_SUBJECTS:
        sub_data = test_data_chest_hrv_eda[test_data_chest_hrv_eda['Subject'] == sub_id]
        sub_data_wrist = test_data_all_wrist[test_data_all_wrist['Subject'] == sub_id]

        sub_data_2_cls = test_data_all_2_cls[test_data_all_2_cls['Subject'] == sub_id]

        if len(sub_data) == 0:
            continue

        X_test_sub_3_cls = sub_data.drop(columns=["Label", "Subject"])
        X_test_sub_2_cls = sub_data_2_cls.drop(columns=["Label", "Subject"])


        y_test_sub_3_cls = sub_data["Label"].values
        y_test_sub_2_cls = sub_data_2_cls["Label"].values
        X_test_sub_2_cls, y_test_sub_2_cls = dataLoading.provide_test_data_fused_2cls(sub_id=sub_id, option="chest", resp=False)

        X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls = dataLoading.provide_test_data_fused(sub_id, option="chest", hrv=True, eda=True, resp=False)
        # Classic Test: RF, CNN, TRANS, LSTM
        #acc_rf, y_pred_rf = smu.predict_model(trained_models_chest_hrv_eda['RF'], X_test_sub_3_cls, y_test_sub_3_cls, 'RF')
        #acc_cnn, y_pred_cnn = smu.predict_model(trained_models_chest_hrv_eda['CNN'], X_test_sub_3_cls, y_test_sub_3_cls, 'CNN')
        #acc_trans, y_pred_trans = smu.predict_model(trained_models_chest_hrv_eda['TRANS'], X_test_sub_3_cls, y_test_sub_3_cls, 'TRANS')
        #acc_lstm, y_pred_lstm = smu.predict_model(trained_models_chest_hrv_eda['LSTM'], X_test_sub_3_cls, y_test_sub_3_cls, 'LSTM')

        acc_rf, y_pred_rf = smu.predict_model(trained_models_chest_hrv_eda['RF'], X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls, 'RF')
        acc_cnn, y_pred_cnn = smu.predict_model(trained_models_chest_hrv_eda['CNN'], X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls, 'CNN')
        acc_trans, y_pred_trans = smu.predict_model(trained_models_chest_hrv_eda['TRANS'], X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls, 'TRANS')
        acc_lstm, y_pred_lstm = smu.predict_model(trained_models_chest_hrv_eda['LSTM'], X_test_sub_chest_hrv_eda_3_cls, y_test_sub_chest_hrv_eda_3_cls, 'LSTM')

        acc_rf_2_cls, y_pred_rf_2_cls = smu.predict_model(RF_2_cls, X_test_sub_2_cls, y_test_sub_2_cls, 'RF')


        xxt3_chest_full, yyt3_chest_full, _ = dataLoading.provide_test_data_concat(sub_id, option="chest", hrv=True, eda=True, resp=True)
        ## CNN specializat RESPIBAN - Full - 3 cls
        acc_multi_cnn_chest_3cls_full, y_pred_multi_cnn_chest_3cls_full, _ = smu.predict_multi_branch_by_vector_count(
            trained_multi_cnn_chest_3cls_full, xxt3_chest_full, yyt3_chest_full)

        ## LSTM specializat RESPIBAN - Full - 3 cls
        acc_multi_lstm_chest_3cls_full, y_pred_multi_lstm_chest_3cls_full, _ = smu.predict_multi_branch_lstm_by_vector_count(
            trained_multi_lstm_chest_3cls_full, xxt3_chest_full, yyt3_chest_full)

        xxt3_chest_hrv_eda, yyt3_chest_hrv_eda, _ = dataLoading.provide_test_data_concat(sub_id, option="chest",hrv=True, eda=True, resp=False)
        #CNN specializat RESPIBAN - hrv/eda - 3 cls
        acc_multi_cnn_chest_3cls_hrv_eda, y_pred_multi_cnn_3_chest_3cls_hrv_eda, raw_pred_multi_cnn_3_chest_3cls_hrv_eda = smu.predict_multi_branch_by_vector_count(trained_multi_cnn_chest_3cls_hrv_eda, xxt3_chest_hrv_eda, yyt3_chest_hrv_eda)

        #LSTM specializat RESPIBAN - hrv/eda - 3 cls
        acc_multi_lstm_chest_3cls_hrv_eda, y_pred_lstm_multi_chest_3cls_hrv_eda, raw_pred_lstm_multi_chest_3cls_hrv_eda = smu.predict_multi_branch_lstm_by_vector_count(trained_multi_lstm_chest_3cls_hrv_eda, xxt3_chest_hrv_eda, yyt3_chest_hrv_eda)

        if skip_some_models:
            print(" SKIPPED SOME MODELS - PREDICTION ")
        else:
            ## RF specializat RESPIBAN - hrv/eda - 3 cls
            acc_multi_rf_chest_3cls_hrv_eda, y_pred_multi_rf_chest_3cls_hrv_eda, _ = smu.predict_multi_rf_independent_branches(trained_multi_rf_chest_3cls_hrv_eda, xxt3_chest_hrv_eda, yyt3_chest_hrv_eda)

            xxt3_wrist_full, yyt3_wrist_full, _ = dataLoading.provide_test_data_concat(sub_id, option="wrist", hrv=True, eda=True, resp=False)
            #CNN specializat Empatica E4 - Full - 3 cls
            acc_multi_cnn_wrist_3cls_full, y_pred_multi_cnn_3_wrist_3cls_full, _ = smu.predict_multi_branch_by_vector_count(trained_multi_cnn_wrist_3cls_full,xxt3_wrist_full,yyt3_wrist_full)

            #LSTM specializat Empatica E4 - full - 3 cls
            acc_multi_lstm_wrist_3cls_full, y_pred_multi_lstm_wrist_3cls_full, _ = smu.predict_multi_branch_lstm_by_vector_count(trained_multi_lstm_wrist_3cls_full, xxt3_wrist_full, yyt3_wrist_full)




            wxxt, wyyt, _ = dataLoading.provide_test_data_concat(sub_id, option="wrist", hrv=True, eda=True, resp=False)
            y_test_sub_wrist_3_cls = sub_data_wrist["Label"].values
            X_test_sub_wrist_3_cls = sub_data_wrist.drop(columns=["Label", "Subject"])
            acc_multi_rf_3cls_wrist, y_pred_multi_rf_3cls_wrist_full, _ = smu.predict_multi_rf_independent_branches(trained_multi_rf_2, wxxt, wyyt)

        print(
            f"\n  Result {sub_id}: RF={acc_rf:.2f}, CNN={acc_cnn:.2f}, Transformer={acc_trans:.2f}, LSTM={acc_lstm:.2f}")

        # Salvăm metricile obținute în listă
        results.append({
            'subject': sub_id,
            'acc_rf': acc_rf,
            'acc_cnn': acc_cnn,
            'acc_transformer': acc_trans,
            'acc_lstm': acc_lstm,
            'acc_rf_2_cls': acc_rf_2_cls
        })
        results_model_fusion.append({
            'subject': sub_id,
            'acc_multi_cnn_chest_3cls_hrv_eda': acc_multi_cnn_chest_3cls_hrv_eda,
            'acc_multi_lstm_chest_3cls_hrv_eda': acc_multi_lstm_chest_3cls_hrv_eda,
            'acc_multi_rf_chest_3cls_hrv_eda': acc_multi_rf_chest_3cls_hrv_eda,
            'acc_multi_rf_3cls_wrist': acc_multi_rf_3cls_wrist,
        })
        results_decision_fusion.append({
            'subject': sub_id,
        })

        # Afișarea matricelor de confuzie aferente subiectului curent
        print(f"  Displaying Confusion Matrices for {sub_id}...")
        skip_plots = True
        if skip_plots:
            print("skipped plots")
        else:
            plot_subject_confusion_matrices(
                sub_id,
                y_test_sub_3_cls,
                y_pred_rf,
                y_pred_cnn,
                y_pred_trans,
                y_pred_lstm,
                class_names
            )
            model_names = ['Multi CNN 3', 'Multi LSTM', 'Chest RF Full', 'Wrist RF Full']
            plot_subject_confusion_matrices(
                sub_id,
                y_test_sub_3_cls,
                y_pred_multi_cnn_3_chest_3cls_hrv_eda,
                y_pred_lstm_multi_chest_3cls_hrv_eda,
                y_pred_multi_rf_chest_3cls_hrv_eda,
                y_pred_multi_rf_chest_3cls_hrv_eda,
                class_names,
                model_names
            )

            model_names = ['Wrist RF Full', 'Wrist RF Full', 'Wrist RF Full', 'Wrist RF Full']
            plot_subject_confusion_matrices(
                sub_id,
                y_test_sub_wrist_3_cls,
                y_pred_multi_rf_3cls_wrist_full,
                y_pred_multi_rf_3cls_wrist_full,
                y_pred_multi_rf_3cls_wrist_full,
                y_pred_multi_rf_3cls_wrist_full,
                class_names,
                model_names
            )
            plting.plot_sub_conf_mat(
                sub_id,
                y_test_sub_2_cls,
                y_pred_rf_2_cls,
                class_names_binary,
                model_name='RF 2 Classes'
            )

    # --- AFISARE REZULTATE FINALE MEDII ---
    df_results = pd.DataFrame(results)
    df_results_model_fusion = pd.DataFrame(results_model_fusion)
    print("\n=== FINAL RESULTS ===")
    print(df_results.to_string(index=False))
    print(df_results_model_fusion.to_string(index=False))

    if not df_results.empty:
        print(f"\nAverage Accuracy on Test Set ({len(TEST_SUBJECTS)} subjects):")
        print(f"RF:  {df_results['acc_rf'].mean():.2f}")
        print(f"CNN: {df_results['acc_cnn'].mean():.2f}")
        print(f"Transformer (TRANS): {df_results['acc_transformer'].mean():.2f}")
        print(f"LSTM: {df_results['acc_lstm'].mean():.2f}")
        print(f"Multi CNN: {df_results_model_fusion['acc_multi_cnn_chest_3cls_hrv_eda'].mean():.2f}")
        print(f"Multi LSTM: {df_results_model_fusion['acc_multi_lstm_chest_3cls_hrv_eda'].mean():.2f}")
        #print(f"Multi RF: {df_results['acc_multi_rf'].mean():.2f}")
        print(f"RF 2 classes: {df_results['acc_rf_2_cls'].mean():.2f}")
    list_raw_preds = [raw_pred_multi_cnn_3_chest_3cls_hrv_eda, raw_pred_lstm_multi_chest_3cls_hrv_eda]
    acc_res, preds_res, raw_res = smu.combine_results_multiple_models(list_raw_preds, yyt3_chest_hrv_eda)
    print(f"Y List chest full: {yyt3_chest_full}")
    print(f"Y List wrist full: {yyt3_wrist_full}")
    #print(f"CNN RAW: {raw_pred_multi_cnn_3_chest_3cls_hrv_eda}")
    #print(f"LSTM RAW: {raw_pred_lstm_multi_chest_3cls_hrv_eda}")
    #print(f"combined RAW: {raw_res}")
    print(f"ACC multi_cnn_3cls_chest_hrv_eda + multi_lstm_3cls_chest_hrv_eda: {acc_res:.2f}")



if __name__ == "__main__":
    main()