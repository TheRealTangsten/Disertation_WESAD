import constants as cnst
import pandas as pd
import numpy as np
import pickle
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import LabelEncoder
import data_loading as dataLoading
import pandas as pd

# Parametrii globali preluați din proiectul tău
SAMPLING_RATE = 700
WINDOW_SIZE_SEC = 120
WINDOW_STEP_SEC = 40
window_size_samples = WINDOW_SIZE_SEC * SAMPLING_RATE
step_size_samples = WINDOW_STEP_SEC * SAMPLING_RATE
DATA_PATH = cnst.path_data

ALL_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
TEST_SUBJECTS = ['S15', 'S16', 'S17']
class_names = ['Baseline', 'Stress', 'Amusement']

chest_df_hrv= []
chest_df_eda= []
chest_df_resp= []

chest_hrv_test_data = []
chest_eda_test_data = []
chest_resp_test_data = []

chest_hrv_train_data = []
chest_eda_train_data = []
chest_resp_train_data = []

chest_hrv_X_train = []
chest_eda_X_train = []
chest_resp_X_train = []


chest_hrv_Y_train = []
chest_eda_Y_train = []
chest_resp_Y_train = []

wrist_hrv_test_data = []
wrist_eda_test_data = []
wrist_hrv_train_data = []
wrist_eda_train_data = []

wrist_hrv_X_train = []
wrist_eda_X_train = []
wrist_hrv_Y_train = []
wrist_eda_Y_train = []

wrist_df_hrv = []
wrist_df_eda = []
def prepare_train_test_data():
    global chest_df_hrv
    global chest_df_eda
    global chest_df_resp

    global chest_hrv_test_data
    global chest_eda_test_data
    global chest_resp_test_data

    global chest_hrv_train_data
    global chest_eda_train_data
    global chest_resp_train_data

    global chest_hrv_X_train
    global chest_eda_X_train
    global chest_resp_X_train

    global chest_hrv_Y_train
    global chest_eda_Y_train
    global chest_resp_Y_train

    global wrist_hrv_test_data
    global wrist_eda_test_data
    global wrist_hrv_train_data
    global wrist_eda_train_data

    global wrist_hrv_X_train
    global wrist_eda_X_train
    global wrist_hrv_Y_train
    global wrist_eda_Y_train

    global wrist_df_hrv
    global wrist_df_eda

    #full_df = load_processed_data(json_type="chest", include_resp=True)
    chest_df_hrv = dataLoading.load_processed_data(json_type="chest", include_hrv=True, include_eda=False,include_resp=False)
    chest_df_eda = dataLoading.load_processed_data(json_type="chest", include_hrv=False, include_eda=True,include_resp=False)
    chest_df_resp = dataLoading.load_processed_data(json_type="chest", include_hrv=False, include_eda=False,include_resp=True)

    wrist_df_hrv = dataLoading.load_processed_data(json_type="wrist", include_hrv=True, include_eda=False,include_resp=False)
    wrist_df_eda = dataLoading.load_processed_data(json_type="wrist", include_hrv=False, include_eda=True,include_resp=False)


    le = LabelEncoder()

    chest_df_hrv['Label'] = le.fit_transform(chest_df_hrv['Label'])
    chest_df_eda['Label'] = le.fit_transform(chest_df_eda['Label'])
    chest_df_resp['Label'] = le.fit_transform(chest_df_resp['Label'])

    wrist_df_hrv['Label'] = le.fit_transform(wrist_df_hrv['Label'])
    wrist_df_eda['Label'] = le.fit_transform(wrist_df_eda['Label'])
    num_classes = len(le.classes_)

    # SPLIT DATE: SUBIECȚI ANTRENARE vs SUBIECȚI TEST
    print(f"\n[INFO] Splitting Data. Test Subjects: {TEST_SUBJECTS}")
    #test_data_all = full_df[full_df['Subject'].isin(TEST_SUBJECTS)].copy()
    #train_data_all = full_df[~full_df['Subject'].isin(TEST_SUBJECTS)].copy()

    chest_hrv_test_data = chest_df_hrv[chest_df_hrv['Subject'].isin(TEST_SUBJECTS)].copy()
    chest_eda_test_data = chest_df_eda[chest_df_eda['Subject'].isin(TEST_SUBJECTS)].copy()
    chest_resp_test_data = chest_df_resp[chest_df_resp['Subject'].isin(TEST_SUBJECTS)].copy()

    wrist_hrv_test_data = wrist_df_hrv[wrist_df_hrv['Subject'].isin(TEST_SUBJECTS)].copy()
    wrist_eda_test_data = wrist_df_eda[wrist_df_eda['Subject'].isin(TEST_SUBJECTS)].copy()

    chest_hrv_train_data = chest_df_hrv[~chest_df_hrv['Subject'].isin(TEST_SUBJECTS)].copy()
    chest_eda_train_data = chest_df_eda[~chest_df_eda['Subject'].isin(TEST_SUBJECTS)].copy()
    chest_resp_train_data = chest_df_resp[~chest_df_resp['Subject'].isin(TEST_SUBJECTS)].copy()

    wrist_hrv_train_data = wrist_df_hrv[~wrist_df_hrv['Subject'].isin(TEST_SUBJECTS)].copy()
    wrist_eda_train_data = wrist_df_eda[~wrist_df_eda['Subject'].isin(TEST_SUBJECTS)].copy()


    # Separăm feature-urile de label/subiect pentru setul de train
    #X_train = train_data_all.drop(columns=["Label", "Subject"])

    chest_hrv_X_train = chest_hrv_train_data.drop(columns=["Label", "Subject"])
    chest_eda_X_train = chest_eda_train_data.drop(columns=["Label", "Subject"])
    chest_resp_X_train = chest_resp_train_data.drop(columns=["Label", "Subject"])

    wrist_hrv_X_train = wrist_hrv_train_data.drop(columns=["Label", "Subject"])
    wrist_eda_X_train = wrist_eda_train_data.drop(columns=["Label", "Subject"])

    #y_train = train_data_all["Label"].values

    chest_hrv_Y_train = chest_hrv_train_data["Label"].values
    chest_eda_Y_train = chest_eda_train_data["Label"].values
    chest_resp_Y_train = chest_resp_train_data["Label"].values

    wrist_hrv_Y_train = wrist_hrv_train_data["Label"].values
    wrist_eda_Y_train = wrist_eda_train_data["Label"].values



    return 0