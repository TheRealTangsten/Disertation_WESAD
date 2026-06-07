import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

# Importurile către arhitecturile tale
import cnn_model as cnn
import transformer_model as transformer
import lstm_model as lstm


def _split_features_by_modality(X_df):
    """
    Funcție ajutătoare privată care separă un DataFrame X în 3 grupuri de caracteristici
    (HRV, EDA, RESP/TEMP) pe baza numelui coloanelor, pregătindu-le pentru Multi-Branch CNN.
    """
    # Cuvinte cheie identice cu cele din data_loading.py
    hrv_keywords = ['HRV', 'ECG_Rate', 'BVP_Rate']
    eda_keywords = ['EDA_', 'SCR_', 'SCL_']
    resp_keywords = ['Respir', 'RRV', 'RESP_']

    col_hrv = [c for c in X_df.columns if any(k in c for k in hrv_keywords)]
    col_eda = [c for c in X_df.columns if any(k in c for k in eda_keywords)]
    col_resp = [c for c in X_df.columns if any(k in c for k in resp_keywords)]

    # Extragem array-urile și le redimensionăm în format 3D (Samples, Features, 1)
    # Dacă o modalitate nu are coloane găsite, returnează None
    x_hrv = X_df[col_hrv].values.reshape(X_df.shape[0], len(col_hrv), 1) if col_hrv else None
    x_eda = X_df[col_eda].values.reshape(X_df.shape[0], len(col_eda), 1) if col_eda else None
    x_resp = X_df[col_resp].values.reshape(X_df.shape[0], len(col_resp), 1) if col_resp else None

    # Obținem formele (shapes) pentru instanțierea straturilor Input de Keras
    shape_hrv = (len(col_hrv), 1) if col_hrv else None
    shape_eda = (len(col_eda), 1) if col_eda else None
    shape_resp = (len(col_resp), 1) if col_resp else None

    inputs_dict = {}
    shapes_dict = {}

    if x_hrv is not None:
        inputs_dict['hrv'] = x_hrv
        shapes_dict['hrv'] = shape_hrv
    if x_eda is not None:
        inputs_dict['eda'] = x_eda
        shapes_dict['eda'] = shape_eda
    if x_resp is not None:
        inputs_dict['resp'] = x_resp
        shapes_dict['resp'] = shape_resp

    return inputs_dict, shapes_dict


def train_model(X_train, y_train, num_classes, model_name):
    """
    Antrenează un singur model specificat prin model_name ('RF', 'CNN', 'TRANS', 'LSTM', 'MULTI_CNN').
    Returnează modelul antrenat.
    """
    print(f"\n[INFO] Starting training for {model_name} on {len(X_train)} samples...")
    BATCH_SIZE = 32

    # 1. Cazul Random Forest (Machine Learning Clasic)
    if model_name == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print(f"[INFO] {model_name} training complete.")
        return model

    # 3. Cazul Modelelor DL Standard (Single-Branch / Fuziune timpurie)
    elif model_name in ['CNN', 'TRANS', 'LSTM']:
        X_train_dl = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train_cat = to_categorical(y_train, num_classes=num_classes)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_dl, y_train_cat))
        train_dataset = train_dataset.shuffle(1000, seed=42, reshuffle_each_iteration=True).batch(BATCH_SIZE,
                                                                                                  drop_remainder=True)

        input_shape = (X_train.shape[1], 1)

        if model_name == 'CNN':
            model = cnn.build_cnn_model(input_shape, num_classes)
        elif model_name == 'TRANS':
            model = transformer.build_transformer_model(input_shape, num_classes)
        elif model_name == 'LSTM':
            model = lstm.build_lstm_model(input_shape, num_classes)

        model.fit(train_dataset, epochs=20, verbose=0)
        print(f"[INFO] {model_name} training complete.")
        return model

    else:
        raise ValueError(
            f"Modelul '{model_name}' nu este recunoscut. Alege dintre: 'RF', 'CNN', 'TRANS', 'LSTM', 'MULTI_CNN'.")


def predict_model(model, X_test, y_test, model_name):
    """
    Realizează predicții pentru un model dat.
    Returnează acuratețea și vectorul de predicții (y_pred).
    """
    BATCH_SIZE = 32
    raw_preds = 0

    # 1. Predicție Random Forest
    if model_name == 'RF':
        y_pred = model.predict(X_test)
        raw_preds = model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)


    # 3. Predicție Deep Learning Standard (Single-Branch)
    elif model_name in ['CNN', 'TRANS', 'LSTM']:
        X_test_dl = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
        test_dataset = tf.data.Dataset.from_tensor_slices(X_test_dl)
        test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)

        probs = model.predict(test_dataset, verbose=0)
        raw_preds = probs

        if len(probs) > 0:
            y_pred = np.argmax(probs, axis=1)
            acc = accuracy_score(y_test, y_pred)
        else:
            y_pred = np.zeros_like(y_test)
            acc = 0.0

    else:
        raise ValueError(f"Modelul '{model_name}' nu este recunoscut.")

    return acc, y_pred, raw_preds







def train_multi_branch_by_vector_count(X_train, y_train, num_classes):
    """
    Antrenează modelul Multi-Branch CNN pe baza unei liste de DataFrame-uri.
    """
    if not X_train or len(X_train) == 0:
        raise ValueError("[ERROR] X_train este gol!")

    num_branches = len(X_train)
    num_samples = X_train[0].shape[0]
    BATCH_SIZE = 32

    print(f"\n[INFO] Starting training for Multi-Branch CNN by vector count...")
    print(f"  -> Detected {num_branches} input modalities/branches.")
    print(f"  -> Total training samples: {num_samples}")

    inputs_list = []
    shapes_dict = {}

    for b_idx in range(num_branches):
        df_branch = X_train[b_idx]
        branch_array = df_branch.values
        num_features = branch_array.shape[1]

        # Redimensionăm în formatul 3D (Samples, Features, 1) pentru Conv1D
        branch_3d = branch_array.reshape(num_samples, num_features, 1)
        inputs_list.append(branch_3d)

        shapes_dict[f'branch_{b_idx}'] = (num_features, 1)
        print(f"     * Branch {b_idx} shape: {(num_features, 1)}")

    y_train_actual = np.array(y_train[0])
    y_train_cat = to_categorical(y_train_actual, num_classes=num_classes)

    # Pentru ANTRENARE: Folosim un dicționar de intrări în Dataset pentru a menține numele ramurilor intacte
    # Cheile trebuie să se potrivească cu numele straturilor Input generate în cnn_model: f"input_branch_{b_idx}"
    input_names_dict = {f"input_branch_{i}": inputs_list[i] for i in range(num_branches)}

    train_dataset = tf.data.Dataset.from_tensor_slices((input_names_dict, y_train_cat))
    train_dataset = train_dataset.shuffle(1000, seed=42, reshuffle_each_iteration=True).batch(BATCH_SIZE,
                                                                                              drop_remainder=True)

    # Construim modelul dinamic
    model = cnn.build_multi_branch_cnn_model(num_classes=num_classes, **shapes_dict)

    # Antrenare
    model.fit(train_dataset, epochs=20, verbose=1)
    print("[INFO] Multi-Branch CNN training complete.")

    return model


def predict_multi_branch_by_vector_count(model, X_test, y_test):
    """
    Realizează predicții primind o listă de DataFrame-uri de test [df_hrv, df_eda, df_resp].
    Evită complet erorile de structură Dataset și retracing-ul TensorFlow.
    """
    if not X_test or len(X_test) == 0:
        raise ValueError("[ERROR] X_test este gol!")

    num_branches = len(X_test)
    num_samples = X_test[0].shape[0]
    BATCH_SIZE = 32

    inputs_list = []

    # Formatăm setul de test în mod identic în array-uri 3D NumPy
    for b_idx in range(num_branches):
        df_branch = X_test[b_idx]
        branch_array = df_branch.values
        num_features = branch_array.shape[1]

        branch_3d = branch_array.reshape(num_samples, num_features, 1)
        inputs_list.append(branch_3d)

    # SOLUȚIE CONCRETĂ: Pasăm direct lista de array-uri NumPy direct în model.predict
    # Keras va mapa automat elementele listei cu cele 3 straturi Input în mod nativ,
    # eliminând problema re-compilării (retracing) indiferent de numărul de eșantioane ale subiectului.
    probs = model.predict(inputs_list, batch_size=BATCH_SIZE, verbose=0)

    # Dacă primul element din y_test are aceeași lungime ca numărul de eșantioane testate, înseamnă că extragem acel sub-vector
    if isinstance(y_test, list) and len(y_test) > 0 and hasattr(y_test[0], '__len__') and len(y_test[0]) == num_samples:
        y_test_arr = np.array(y_test[0])
    # În cazul în care în bucla din main ai pasat direct array-ul curat al subiectului (ex: y_test_sub)
    else:
        y_test_arr = np.array(y_test)
    if len(probs) > 0:
        y_pred = np.argmax(probs, axis=1)
        print(f"Multi CNN: y_test_len:{y_test_arr.shape} | y_pred_len:{len(y_pred)}")
        acc = accuracy_score(y_test_arr, y_pred)
    else:
        y_pred = np.zeros_like(y_test_arr)
        acc = 0.0

    return acc, y_pred, probs


def train_multi_branch_lstm_by_vector_count(X_list, y_list, num_classes, epochs=30, batch_size=32,
                                            class_weights_dict=None):
    """
    Antrenează modelul LSTM multimodal utilizând ieșirea directă de tip listă (xt, yt).
    """
    print(f"\n[INFO] Pregătire date Train pentru Multi-Branch LSTM...")

    # 1. Extrage etichetele globale din prima ramură (sunt identice pentru toate ramurile)
    y_train = y_list[0]
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    y_train_cat = to_categorical(y_train, num_classes=num_classes)

    prepared_inputs = []
    branch_shapes = []

    # 2. Reshaping automat din 2D în 3D (samples, timesteps, 1) pentru fiecare ramură din listă
    for i, X_branch in enumerate(X_list):
        if hasattr(X_branch, 'values'):
            X_branch = X_branch.values

        # Transformăm în formatul cerut de LSTM: (baze_date, caracteristici, 1 canal)
        X_3d = X_branch.reshape(X_branch.shape[0], X_branch.shape[1], 1)
        prepared_inputs.append(X_3d)

        # Salvăm configurația dimensională pentru construcția rețelei
        branch_shapes.append((X_branch.shape[1], 1))
        print(f"  -> Ramura {i} procesată cu dimensiunea de input: {branch_shapes[-1]}")

    # 3. Construirea automată a modelului pe baza structurii listei trimise
    model = lstm.build_multi_branch_lstm_model(num_classes, branch_shapes)

    # 4. Rularea antrenării
    print("  -> Se începe antrenarea modelului Multi-Branch LSTM...")
    model.fit(
        x=prepared_inputs,
        y=y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict,
        verbose=1
    )

    return model

def predict_multi_branch_lstm_by_vector_count(model, X_list, y_list):
    """
    Efectuează predicții și returnează acuratețea și etichetele prezise pentru setul de test (xxt, yyt).
    """
    # 1. Reshaping date de test în format 3D (samples, timesteps, 1)
    prepared_inputs = []
    for X_branch in X_list:
        if hasattr(X_branch, 'values'):
            X_branch = X_branch.values
        X_3d = X_branch.reshape(X_branch.shape[0], X_branch.shape[1], 1)
        prepared_inputs.append(X_3d)

    # 2. Generare predicții brute (probabilități)
    probs = model.predict(prepared_inputs, verbose=0)

    if len(probs) > 0:
        y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = np.zeros(len(X_list[0]))

    # 3. Preluarea etichetelor reale de test pentru evaluare
    y_test = y_list[0]
    if hasattr(y_test, 'values'):
        y_test = y_test.values

    # Calcul acuratețe scurt și curat
    acc = accuracy_score(y_test, y_pred)

    return acc, y_pred, probs



def train_multi_rf_independent_branches(X_list, y_list, num_classes=3, n_estimators=100, class_weights_dict=None):
    """
    Antrenează un model Random Forest separat pentru fiecare vector de intrare (ramură).
    Returnează o listă de modele antrenate.
    """
    print(f"\n[INFO] Pregătire date Train pentru Multi-RF (Independent pe ramuri)...")

    # 1. Extragem etichetele (comune pentru toate ramurile)
    y_train = y_list[0]
    if hasattr(y_train, 'values'):
        y_train = y_train.values

    cw = class_weights_dict if class_weights_dict is not None else 'balanced'

    trained_rf_models = []

    # 2. Iterăm prin fiecare ramură de date și antrenăm un RF dedicat
    for i, X_branch in enumerate(X_list):
        if hasattr(X_branch, 'values'):
            X_branch = X_branch.values

        print(f"  -> Antrenare RF independent pentru ramura {i} (Dimensiune: {X_branch.shape})")

        # Inițializăm și antrenăm modelul specific acestei modalități
        model = RandomForestClassifier(n_estimators=n_estimators, class_weight=cw, random_state=42)
        model.fit(X_branch, y_train)

        # Salvăm modelul în lista de modele
        trained_rf_models.append(model)

    return trained_rf_models


def predict_multi_rf_independent_branches(models_list, X_list, y_list):
    """
    Efectuează predicții folosind lista de modele RF antrenate (unul per ramură).
    Folosește metoda 'Soft Voting' (media probabilităților) pentru a lua decizia finală.
    """
    # Etichetele reale pentru calculul acurateței
    y_test = y_list[0]
    if hasattr(y_test, 'values'):
        y_test = y_test.values

    num_samples = X_list[0].shape[0] if not hasattr(X_list[0], 'values') else X_list[0].values.shape[0]
    num_classes = len(models_list[0].classes_)

    # Inițializăm o matrice zero pentru a aduna probabilitățile de la fiecare model
    # Dimensiune: (număr_subiecți_test, număr_clase)
    summed_probs = np.zeros((num_samples, num_classes))

    # Iterăm simultan prin modele și prin ramurile de test corespunzătoare
    for model, X_branch in zip(models_list, X_list):
        if hasattr(X_branch, 'values'):
            X_branch = X_branch.values

        # Obținem probabilitățile prezise de acest model specific (ex. doar din HRV)
        probs = model.predict_proba(X_branch)

        # Adunăm probabilitățile
        summed_probs += probs

    # Calculăm media probabilităților împărțind la numărul de modele (ramuri)
    avg_probs = summed_probs / len(models_list)

    # Clasa finală este cea cu probabilitatea medie maximă
    y_pred = np.argmax(avg_probs, axis=1)

    # Calculăm acuratețea
    acc = accuracy_score(y_test, y_pred)

    return acc, y_pred, avg_probs


def combine_results_multiple_models(list_raw_preds, y_list):
    raws_avg = sum(list_raw_preds)/len(list_raw_preds)
    preds = np.argmax(raws_avg, axis=1)
    y_intermediary = np.asarray(y_list)
    print(f"{y_intermediary.shape} - {len(y_intermediary.shape)}")
    y_test = y_list if len(y_intermediary.shape) == 1 else y_list[0]
    print(f"Y Test: {y_test}")
    print(f"Y List: {y_list}")
    acc = accuracy_score(y_test, preds)

    return acc, preds, raws_avg

def combine_results_single_3cls_plus_2cls(preds_3cls, preds_2cls, y_list):
    raw_3cls = np.asarray(preds_3cls)
    raw_2cls =  np.asarray(preds_2cls)
    positive_rating = preds_2cls[:, 0:1]
    negative_rating = preds_2cls[:, 1:2]
    raw_3cls[:, [0,2] ] += positive_rating
    raw_3cls[:, [1] ] += negative_rating
    raws_avg = raw_3cls/2
    preds = np.argmax(raws_avg, axis=1)
    y_intermediary = np.asarray(y_list)
    print(f"{y_intermediary.shape} - {len(y_intermediary.shape)}")
    y_test = y_list if len(y_intermediary.shape) == 1 else y_list[0]
    print(f"Y Test: {y_test}")
    print(f"Y List: {y_list}")
    acc = accuracy_score(y_test, preds)

    return acc, preds, raws_avg
