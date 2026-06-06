import constants as cnst
import pandas as pd
import numpy as np
import pickle
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import LabelEncoder
import data_loading_constants as dlc
import data_loading_constants_2classes as dlc2

# Parametrii globali preluați din proiectul tău
SAMPLING_RATE = 700
WINDOW_SIZE_SEC = 120
WINDOW_STEP_SEC = 40
window_size_samples = WINDOW_SIZE_SEC * SAMPLING_RATE
step_size_samples = WINDOW_STEP_SEC * SAMPLING_RATE
DATA_PATH = cnst.path_data

init_call_provide_test_train = True
init_call_provide_test_train_2cls = True


ALL_SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
TEST_SUBJECTS = ['S15', 'S16', 'S17']
class_names = ['Baseline', 'Stress', 'Amusement']


# =====================================================================
# 1. FUNCȚIILE DE EXTRAGERE (Mutate aici din WESAD_Comparativ.py)
# =====================================================================

def extract_features_from_subject(subject_id):
    print(f"Loading data for {subject_id} including Respiration...")
    try:
        with open(f"{DATA_PATH}{subject_id}/{subject_id}.pkl", 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Could not load data for {subject_id}: {e}")
        return None

    ecg_signal = data['signal']['chest']['ECG']
    eda_signal = data['signal']['chest']['EDA'].flatten()
    resp_signal = data['signal']['chest']['Resp'].flatten()  # Semnalul brut de respirație
    labels = data['label']

    try:
        # Procesare ECG și EDA (existente)
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=SAMPLING_RATE)
        _, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=SAMPLING_RATE)
        rpeaks_indices = rpeaks['ECG_R_Peaks']
        eda_processed, _ = nk.eda_process(eda_signal, sampling_rate=SAMPLING_RATE)

        # --- NEW: Procesare Respirație ---
        # Curățăm semnalul și extragem trăsăturile specifice (Rhythm, Rate)
        resp_processed, _ = nk.rsp_process(resp_signal, sampling_rate=SAMPLING_RATE)
    except Exception as e:
        print(f"  Signal processing failed for {subject_id}: {e}")
        return None

    features_list = []

    for start in range(0, len(ecg_signal) - window_size_samples, step_size_samples):
        end = start + window_size_samples
        if end > len(labels): break

        window_labels = labels[start:end]
        most_common_label = np.bincount(window_labels).argmax()
        if most_common_label not in [1, 2, 3]: continue

        peaks_in_window = rpeaks_indices[(rpeaks_indices >= start) & (rpeaks_indices < end)] - start

        if len(peaks_in_window) > 3:
            try:
                # 1. HRV Features (Existente)
                peaks_df = pd.DataFrame({"ECG_R_Peaks": np.zeros(window_size_samples, dtype=bool)})
                peaks_df.loc[peaks_in_window, "ECG_R_Peaks"] = True
                hrv = nk.hrv(peaks_df, sampling_rate=SAMPLING_RATE, show=False)
                hrv_row = hrv.select_dtypes(include=[np.number]).iloc[0].to_dict()

                # 2. EDA Features (Existente)
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

                # --- 3. RESP Features (NEW) ---
                resp_window = resp_processed.iloc[start:end]
                resp_rate_val = resp_window['RSP_Rate'].fillna(method='ffill').fillna(method='bfill').mean()
                resp_amp_val = resp_window['RSP_Amplitude'].fillna(0).mean()

                resp_feats = {
                    'RESP_Rate_Mean': resp_rate_val if not np.isnan(resp_rate_val) else 0,
                    'RESP_Amplitude_Mean': resp_amp_val if not np.isnan(resp_amp_val) else 0,
                    'RESP_Std': resp_window['RSP_Clean'].std() if 'RSP_Clean' in resp_window else 0
                }

                fused_row = {**hrv_row, **eda_feats, **resp_feats}
                fused_row["Label"] = most_common_label
                fused_row["Subject"] = subject_id
                features_list.append(fused_row)

            except Exception:
                continue

    if not features_list:
        print("FUCK MICROSOFT")
        return None
    df = pd.DataFrame(features_list)

    # handling valori invalide

    # valori infinite inlocuite cu NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Inlocuire valori NaN cu mediana ( daca exista )
    df = df.fillna(df.median(numeric_only=True))
    # Inlocuire colana NaN cu 0
    df = df.fillna(0)

    # Normalizare per-subiect
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

    print(f"Returning DF for {subject_id} succesfuly I think")
    return df


def extract_wrist_features_from_subject(subject_id):
    print(f"Loading and normalizing wrist data for {subject_id}...")
    try:
        with open(f"{DATA_PATH}{subject_id}/{subject_id}.pkl", 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Could not load data for {subject_id}: {e}")
        return None

    # Frecventele de esantionare (Sampling Rates) specifice bratarii Empatica E4 din WESAD
    BVP_SR = 64
    EDA_WRIST_SR = 4
    LABEL_SR = 700  # Etichetele raman mereu la 700Hz in formatul original

    # Extragem semnalele
    bvp_signal = data['signal']['wrist']['BVP'].flatten()
    eda_wrist_signal = data['signal']['wrist']['EDA'].flatten()
    labels = data['label']

    try:
        # Procesam BVP ca semnal PPG
        bvp_cleaned, info_bvp = nk.ppg_process(bvp_signal, sampling_rate=BVP_SR)
        # Procesam EDA de la incheietura
        eda_processed, _ = nk.eda_process(eda_wrist_signal, sampling_rate=EDA_WRIST_SR)
    except Exception as e:
        print(f"  Signal processing failed for {subject_id}: {e}")
        return None

    features_list = []

    # Calculam durata totala in secunde bazat pe lungimea etichetelor
    total_seconds = len(labels) // LABEL_SR

    # Iteram prin ferestre bazandu-ne pe secunde pentru a sincroniza ratele de esantionare
    for start_sec in range(0, total_seconds - WINDOW_SIZE_SEC, WINDOW_STEP_SEC):
        end_sec = start_sec + WINDOW_SIZE_SEC

        # 1. Extragem eticheta ferestrei (700 Hz)
        start_label = start_sec * LABEL_SR
        end_label = end_sec * LABEL_SR

        if end_label > len(labels): break

        window_labels = labels[start_label:end_label]
        most_common_label = np.bincount(window_labels).argmax()

        # 1=Baseline, 2=Stress, 3=Amusement
        if most_common_label not in [1, 2, 3]:
            continue

        # 2. Decupam fereastra pentru BVP (64 Hz)
        start_bvp = start_sec * BVP_SR
        end_bvp = end_sec * BVP_SR
        bvp_window = bvp_cleaned.iloc[start_bvp:end_bvp]

        # 3. Decupam fereastra pentru EDA (4 Hz)
        start_eda = start_sec * EDA_WRIST_SR
        end_eda = end_sec * EDA_WRIST_SR
        eda_window = eda_processed.iloc[start_eda:end_eda]

        # Cautam varfurile in fereastra curenta BVP pentru calculul HRV
        # Coloana generata de neurokit2 pentru varfuri PPG se numeste 'PPG_Peaks'
        peaks_in_window = bvp_window[bvp_window['PPG_Peaks'] == 1].index

        # Minim 3 batai de inima pentru a putea calcula HRV
        if len(peaks_in_window) > 3:
            try:
                # --- A. Extragere Trasaturi BVP (HRV) ---
                window_len_bvp = end_bvp - start_bvp
                peaks_df = pd.DataFrame({"PPG_Peaks": np.zeros(window_len_bvp, dtype=bool)})

                # Aliniem indicii varfurilor relativ la inceputul ferestrei
                relative_peaks = peaks_in_window - start_bvp
                valid_peaks = relative_peaks[(relative_peaks >= 0) & (relative_peaks < window_len_bvp)]
                peaks_df.loc[valid_peaks, "PPG_Peaks"] = True

                # Calculam HRV folosind varfurile PPG
                hrv = nk.hrv(peaks_df, sampling_rate=BVP_SR, show=False)
                hrv_numeric = hrv.select_dtypes(include=[np.number])
                hrv_row = hrv_numeric.iloc[0].to_dict()

                # --- B. Extragere Trasaturi EDA (Wrist) ---
                eda_feats = {
                    'EDA_Wrist_Mean': eda_window['EDA_Clean'].mean(),
                    'EDA_Wrist_Std': eda_window['EDA_Clean'].std(),
                    'EDA_Wrist_Tonic_Mean': eda_window['EDA_Tonic'].mean(),
                    'EDA_Wrist_Phasic_Mean': eda_window['EDA_Phasic'].mean(),
                    'EDA_Wrist_Phasic_Std': eda_window['EDA_Phasic'].std(),
                    'EDA_Wrist_Min': eda_window['EDA_Clean'].min(),
                    'EDA_Wrist_Max': eda_window['EDA_Clean'].max()
                }

                # Imbinam trasaturile
                fused_row = {**hrv_row, **eda_feats}
                fused_row["Label"] = most_common_label
                fused_row["Subject"] = subject_id
                features_list.append(fused_row)

            except Exception:
                continue  # Trecem peste fereastra daca apare o eroare la extragere

    if not features_list:
        return None

    df = pd.DataFrame(features_list)

    # --- Gestionarea valorilor invalide ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(0)

    # --- Normalizare per-subiect ---
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


# =====================================================================
# 2. NOUA FUNCȚIE DE GENERARE ȘI SALVARE .JSON
# =====================================================================

def preprocess_and_save_to_json(all_subjects, output_dir="Jsons"):
    """
    Trece prin toți subiecții, extrage trăsăturile (piept și încheietură),
    și salvează dataframe-urile rezultate în două fișiere JSON.
    """
    print("\n=== STARTING DATA PREPROCESSING AND EXPORT ===")

    chest_data_frames = []
    wrist_data_frames = []

    for sub_id in all_subjects:
        print(f"\n---> Processing subject: {sub_id}")

        # Extragere date Piept (Chest)
        df_chest = extract_features_from_subject(sub_id)
        if df_chest is not None:
            chest_data_frames.append(df_chest)
        else:
            print(f"  [!] Skipped CHEST data for {sub_id}")

        # Extragere date Încheietură (Wrist)
        df_wrist = extract_wrist_features_from_subject(sub_id)
        if df_wrist is not None:
            wrist_data_frames.append(df_wrist)
        else:
            print(f"  [!] Skipped WRIST data for {sub_id}")

    # -----------------------------------------
    # Salvare CHEST.json
    # -----------------------------------------
    if chest_data_frames:
        full_chest_df = pd.concat(chest_data_frames, ignore_index=True)
        chest_file_path = os.path.join(output_dir, "chest.json")

        # Salvăm ca array de obiecte JSON (orient='records')
        full_chest_df.to_json(chest_file_path, orient="records", indent=4)
        print(f"\n[SUCCESS] Saved CHEST data ({full_chest_df.shape[0]} samples) to {chest_file_path}")
    else:
        print("\n[ERROR] No CHEST data could be processed.")

    # -----------------------------------------
    # Salvare WRIST.json
    # -----------------------------------------
    if wrist_data_frames:
        full_wrist_df = pd.concat(wrist_data_frames, ignore_index=True)
        wrist_file_path = os.path.join(output_dir, "wrist.json")

        full_wrist_df.to_json(wrist_file_path, orient="records", indent=4)
        print(f"[SUCCESS] Saved WRIST data ({full_wrist_df.shape[0]} samples) to {wrist_file_path}")
    else:
        print("[ERROR] No WRIST data could be processed.")

    print("=== EXPORT COMPLETE ===")


def load_processed_data(json_type="chest", folder="Jsons",
                        include_hrv=True, include_eda=True, include_resp=True):
    """
    Citește un fișier JSON (wrist sau chest) și selectează trăsăturile dorite.
    Returnează: X (features), y (labels)
    """
    file_path = os.path.join(folder, f"{json_type}.json")

    if not os.path.exists(file_path):
        print(f"[ERORARE] Fișierul {file_path} nu există!")
        return None, None

    print(f"[INFO] Încărcare date din {file_path}...")
    df = pd.read_json(file_path, orient="records")

    # Identificăm coloanele de bază care nu sunt features (Label și Subject)
    base_cols = ['Label', 'Subject']

    # Definim grupurile de trăsături bazat pe prefixele/numele din funcțiile tale de extracție
    # Notă: Aceste cuvinte cheie trebuie să se regăsească în numele coloanelor generate de NeuroKit2
    hrv_keywords = ['HRV', 'ECG_Rate', 'BVP_Rate']
    eda_keywords = ['EDA_', 'SCR_', 'SCL_']
    resp_keywords = ['Respir', 'RRV', 'RESP_']

    selected_features = []
    if json_type == "wrist":
        include_resp = False
    # Logica de filtrare
    for col in df.columns:

        if col in base_cols:
            selected_features.append(col)
            continue

        keep = False
        if include_hrv and any(key in col for key in hrv_keywords):
            keep = True
        if include_eda and any(key in col for key in eda_keywords):
            keep = True
        if include_resp and any(key in col for key in resp_keywords):
            keep = True

        # Dacă este o coloană de tip feature care nu se încadrează în categorii,
        # dar vrem să o păstrăm (ex: trăsături statistice simple)
        if keep:
            selected_features.append(col)

    # Dacă nu am selectat nimic specific (sau toate sunt True), returnăm toate coloanele de tip feature

    final_df = df[selected_features]

    print(f"[SUCCESS] Date încărcate: {final_df.shape[0]} rânduri, {final_df.shape[1]} coloane totale.")
    return final_df


def load_processed_data_binary(json_type="chest", folder="Jsons",
                               include_hrv=True, include_eda=True, include_resp=True):
    """
    Citește un fișier JSON (wrist sau chest), selectează trăsăturile specificate
    și transformă problema într-una de clasificare binară:
    Baseline (1) și Amusement (3) devin Non-Stress (0), iar Stress (2) devine Stress (1).
    """
    file_path = os.path.join(folder, f"{json_type}.json")

    if not os.path.exists(file_path):
        print(f"[EROARE] Fișierul {file_path} nu există!")
        return None

    print(f"[INFO] Încărcare date din {file_path} pentru clasificare binară (Stress vs Non-Stress)...")
    df = pd.read_json(file_path, orient="records")

    # --- RECTIFICARE MAPARE CLASE ---
    # În JSON-ul brut: 1 = Baseline, 2 = Stress, 3 = Amusement
    # Noua logică:
    # 1 (Baseline)   -> 0 (Non-Stress)
    # 3 (Amusement)  -> 0 (Non-Stress)
    # 2 (Stress)     -> 1 (Stress)
    mapping = {1: 0, 3: 0, 2: 1}

    df['Label'] = df['Label'].map(mapping)

    # Identificăm coloanele de bază care nu reprezintă trăsături semnal
    base_cols = ['Label', 'Subject']

    # Definim cuvintele cheie pentru filtrarea semnalelor
    hrv_keywords = ['HRV', 'ECG_Rate', 'BVP_Rate']
    eda_keywords = ['EDA_', 'SCR_', 'SCL_']
    resp_keywords = ['Respir', 'RRV', 'RESP_']

    selected_features = []

    if json_type == "wrist":
        include_resp = False

    # Logica de filtrare a coloanelor pe baza argumentelor funcției
    for col in df.columns:
        if col in base_cols:
            selected_features.append(col)
            continue

        keep = False
        if include_hrv and any(key in col for key in hrv_keywords):
            keep = True
        if include_eda and any(key in col for key in eda_keywords):
            keep = True
        if include_resp and any(key in col for key in resp_keywords):
            keep = True

        if keep:
            selected_features.append(col)

    final_df = df[selected_features]

    print(f"[SUCCESS] Date binare încărcate: {final_df.shape[0]} rânduri, {final_df.shape[1]} coloane totale.")
    return final_df


# =====================================================================
# 3. LOAD CACHED DATA
# =====================================================================
def provide_train_data_concat(option = "chest", hrv = True, eda = True, resp = True):
    # option = chest / wrist
    global init_call_provide_test_train
    if init_call_provide_test_train == True:
        init_call_provide_test_train = False
        dlc.prepare_train_test_data()

    nr_datas = 0
    return_X_train_data = []
    return_Y_train_data = []
    if option == "chest":
        if hrv == True:
            return_X_train_data.append(dlc.chest_hrv_X_train)
            return_Y_train_data.append(dlc.chest_hrv_Y_train)
            nr_datas = nr_datas + 1
        if eda == True:
            return_X_train_data.append(dlc.chest_eda_X_train)
            return_Y_train_data.append(dlc.chest_eda_Y_train)
            nr_datas = nr_datas + 1
        if resp == True:
            return_X_train_data.append(dlc.chest_resp_X_train)
            return_Y_train_data.append(dlc.chest_resp_Y_train)
            nr_datas = nr_datas + 1

    if option == "wrist":
        resp = False

        if hrv == True:
            return_X_train_data.append(dlc.wrist_hrv_X_train)
            return_Y_train_data.append(dlc.wrist_hrv_Y_train)
            nr_datas = nr_datas + 1
        if eda == True:
            return_X_train_data.append(dlc.wrist_eda_X_train)
            return_Y_train_data.append(dlc.wrist_eda_Y_train)
            nr_datas = nr_datas + 1

    return return_X_train_data, return_Y_train_data, nr_datas

def provide_test_data_concat(sub_id, option = "chest", hrv = True, eda = True, resp = True):

    global init_call_provide_test_train
    if init_call_provide_test_train == True:
        init_call_provide_test_train = False
        dlc.prepare_train_test_data()

    return_X_concat = []
    return_Y_concat = []
    nr_datas = 0
    if option == "chest":
        if hrv  == True:
            sub_chest_hrv_test_data = dlc.chest_hrv_test_data[dlc.chest_hrv_test_data['Subject'] == sub_id]
            X_sub_chest_hrv_test_data = sub_chest_hrv_test_data.drop(columns=["Label", "Subject"])
            Y_sub_chest_hrv_test_data = sub_chest_hrv_test_data["Label"].values
            return_X_concat.append(X_sub_chest_hrv_test_data)
            return_Y_concat.append(Y_sub_chest_hrv_test_data)
            nr_datas = nr_datas + 1

        if eda == True:
            sub_chest_eda_test_data = dlc.chest_eda_test_data[dlc.chest_eda_test_data['Subject'] == sub_id]
            X_sub_chest_eda_test_data = sub_chest_eda_test_data.drop(columns=["Label", "Subject"])
            Y_sub_chest_eda_test_data = sub_chest_eda_test_data["Label"].values
            return_X_concat.append(X_sub_chest_eda_test_data)
            return_Y_concat.append(Y_sub_chest_eda_test_data)
            nr_datas = nr_datas + 1

        if resp == True:
            sub_chest_resp_test_data = dlc.chest_resp_test_data[dlc.chest_resp_test_data['Subject'] == sub_id]
            X_sub_chest_resp_test_data = sub_chest_resp_test_data.drop(columns=["Label", "Subject"])
            Y_sub_chest_resp_test_data = sub_chest_resp_test_data["Label"].values
            return_X_concat.append(X_sub_chest_resp_test_data)
            return_Y_concat.append(Y_sub_chest_resp_test_data)
            nr_datas = nr_datas + 1

    if option == "wrist":
        if hrv == True:
            sub_wrist_hrv_test_data = dlc.wrist_hrv_test_data[dlc.wrist_hrv_test_data['Subject'] == sub_id]
            X_sub_wrist_hrv_test_data = sub_wrist_hrv_test_data.drop(columns=["Label", "Subject"])
            Y_sub_wrist_hrv_test_data = sub_wrist_hrv_test_data["Label"].values
            return_X_concat.append(X_sub_wrist_hrv_test_data)
            return_Y_concat.append(Y_sub_wrist_hrv_test_data)
            nr_datas = nr_datas + 1

        if eda == True:
            sub_wrist_eda_test_data = dlc.wrist_eda_test_data[dlc.wrist_eda_test_data['Subject'] == sub_id]
            X_sub_wrist_eda_test_data = sub_wrist_eda_test_data.drop(columns=["Label", "Subject"])
            Y_sub_wrist_eda_test_data = sub_wrist_eda_test_data["Label"].values
            return_X_concat.append(X_sub_wrist_eda_test_data)
            return_Y_concat.append(Y_sub_wrist_eda_test_data)
            nr_datas = nr_datas + 1



    return return_X_concat, return_Y_concat, nr_datas

def provide_train_data_concat_2cls(option = "chest", hrv = True, eda = True, resp = True):
    # option = chest / wrist
    global init_call_provide_test_train_2cls
    if init_call_provide_test_train_2cls == True:
        init_call_provide_test_train_2cls = False
        dlc2.prepare_train_test_data_2classes()

    nr_datas = 0
    return_X_train_data = []
    return_Y_train_data = []
    if option == "chest":
        if hrv == True:
            return_X_train_data.append(dlc2.chest_hrv_X_train_2cls)
            return_Y_train_data.append(dlc2.chest_hrv_Y_train_2cls)
            nr_datas = nr_datas + 1
        if eda == True:
            return_X_train_data.append(dlc2.chest_eda_X_train_2cls)
            return_Y_train_data.append(dlc2.chest_eda_Y_train_2cls)
            nr_datas = nr_datas + 1
        if resp == True:
            return_X_train_data.append(dlc2.chest_resp_X_train_2cls)
            return_Y_train_data.append(dlc2.chest_resp_Y_train_2cls)
            nr_datas = nr_datas + 1

    if option == "wrist":
        resp = False

        if hrv == True:
            return_X_train_data.append(dlc2.wrist_hrv_X_train_2cls)
            return_Y_train_data.append(dlc2.wrist_hrv_Y_train_2cls)
            nr_datas = nr_datas + 1
        if eda == True:
            return_X_train_data.append(dlc2.wrist_eda_X_train_2cls)
            return_Y_train_data.append(dlc2.wrist_eda_Y_train_2cls)
            nr_datas = nr_datas + 1

    return return_X_train_data, return_Y_train_data, nr_datas

def provide_test_data_concat_2cls(sub_id, option = "chest", hrv = True, eda = True, resp = True):

    global init_call_provide_test_train_2cls
    if init_call_provide_test_train_2cls == True:
        init_call_provide_test_train_2cls = False
        dlc2.prepare_train_test_data_2classes()

    return_X_concat = []
    return_Y_concat = []
    nr_datas = 0
    if option == "chest":
        if hrv  == True:
            sub_chest_hrv_test_data = dlc2.chest_hrv_test_data_2cls[dlc2.chest_hrv_test_data_2cls['Subject'] == sub_id]
            X_sub_chest_hrv_test_data = sub_chest_hrv_test_data.drop(columns=["Label", "Subject"])
            Y_sub_chest_hrv_test_data = sub_chest_hrv_test_data["Label"].values
            return_X_concat.append(X_sub_chest_hrv_test_data)
            return_Y_concat.append(Y_sub_chest_hrv_test_data)
            nr_datas = nr_datas + 1

        if eda == True:
            sub_chest_eda_test_data = dlc2.chest_eda_test_data_2cls[dlc2.chest_eda_test_data_2cls['Subject'] == sub_id]
            X_sub_chest_eda_test_data = sub_chest_eda_test_data.drop(columns=["Label", "Subject"])
            Y_sub_chest_eda_test_data = sub_chest_eda_test_data["Label"].values
            return_X_concat.append(X_sub_chest_eda_test_data)
            return_Y_concat.append(Y_sub_chest_eda_test_data)
            nr_datas = nr_datas + 1

        if resp == True:
            sub_chest_resp_test_data = dlc2.chest_resp_test_data_2cls[dlc2.chest_resp_test_data_2cls['Subject'] == sub_id]
            X_sub_chest_resp_test_data = sub_chest_resp_test_data.drop(columns=["Label", "Subject"])
            Y_sub_chest_resp_test_data = sub_chest_resp_test_data["Label"].values
            return_X_concat.append(X_sub_chest_resp_test_data)
            return_Y_concat.append(Y_sub_chest_resp_test_data)
            nr_datas = nr_datas + 1

    if option == "wrist":
        if hrv == True:
            sub_wrist_hrv_test_data = dlc2.wrist_hrv_test_data_2cls[dlc2.wrist_hrv_test_data_2cls['Subject'] == sub_id]
            X_sub_wrist_hrv_test_data = sub_wrist_hrv_test_data.drop(columns=["Label", "Subject"])
            Y_sub_wrist_hrv_test_data = sub_wrist_hrv_test_data["Label"].values
            return_X_concat.append(X_sub_wrist_hrv_test_data)
            return_Y_concat.append(Y_sub_wrist_hrv_test_data)
            nr_datas = nr_datas + 1

        if eda == True:
            sub_wrist_eda_test_data = dlc2.wrist_eda_test_data_2cls[dlc2.wrist_eda_test_data_2cls['Subject'] == sub_id]
            X_sub_wrist_eda_test_data = sub_wrist_eda_test_data.drop(columns=["Label", "Subject"])
            Y_sub_wrist_eda_test_data = sub_wrist_eda_test_data["Label"].values
            return_X_concat.append(X_sub_wrist_eda_test_data)
            return_Y_concat.append(Y_sub_wrist_eda_test_data)
            nr_datas = nr_datas + 1



    return return_X_concat, return_Y_concat, nr_datas

def provide_train_data_fused(option = "chest", hrv = True, eda = True, resp = True):
    df = load_processed_data(json_type=option,include_hrv=hrv, include_eda=eda, include_resp=resp)

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    num_classes_3 = len(le.classes_)

    print(f"\n[INFO] Splitting Data. Test Subjects: {TEST_SUBJECTS}")

    train_data = df[~df['Subject'].isin(TEST_SUBJECTS)].copy()

    X_train = train_data.drop(columns=["Label", "Subject"])
    y_train = train_data["Label"].values


    return X_train, y_train, num_classes_3

def provide_test_data_fused(sub_id, option = "chest", hrv = True, eda = True, resp = True):
    df = load_processed_data(json_type=option, include_hrv=hrv, include_eda=eda, include_resp=resp)

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    num_classes_3 = len(le.classes_)
    test_data = df[df['Subject'].isin(TEST_SUBJECTS)].copy()

    sub_data = test_data[test_data['Subject'] == sub_id]

    X_test = sub_data.drop(columns=["Label", "Subject"])

    y_test = sub_data["Label"].values

    return X_test, y_test

def provide_train_data_fused_2cls(option = "chest", hrv = True, eda = True, resp = True):
    df = load_processed_data_binary(json_type=option,include_hrv=hrv, include_eda=eda, include_resp=resp)

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    num_classes_3 = len(le.classes_)

    print(f"\n[INFO] Splitting Data. Test Subjects: {TEST_SUBJECTS}")

    train_data = df[~df['Subject'].isin(TEST_SUBJECTS)].copy()

    X_train = train_data.drop(columns=["Label", "Subject"])
    y_train = train_data["Label"].values


    return X_train, y_train, num_classes_3

def provide_test_data_fused_2cls(sub_id, option = "chest", hrv = True, eda = True, resp = True):
    df = load_processed_data_binary(json_type=option, include_hrv=hrv, include_eda=eda, include_resp=resp)

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    num_classes_3 = len(le.classes_)
    test_data = df[df['Subject'].isin(TEST_SUBJECTS)].copy()

    sub_data = test_data[test_data['Subject'] == sub_id]

    X_test = sub_data.drop(columns=["Label", "Subject"])

    y_test = sub_data["Label"].values

    return X_test, y_test



