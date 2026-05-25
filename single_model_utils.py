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

    # 1. Predicție Random Forest
    if model_name == 'RF':
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    # 3. Predicție Deep Learning Standard (Single-Branch)
    elif model_name in ['CNN', 'TRANS', 'LSTM']:
        X_test_dl = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
        test_dataset = tf.data.Dataset.from_tensor_slices(X_test_dl)
        test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)

        probs = model.predict(test_dataset, verbose=0)

        if len(probs) > 0:
            y_pred = np.argmax(probs, axis=1)
            acc = accuracy_score(y_test, y_pred)
        else:
            y_pred = np.zeros_like(y_test)
            acc = 0.0

    else:
        raise ValueError(f"Modelul '{model_name}' nu este recunoscut.")

    return acc, y_pred


import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import cnn_model as cnn


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

    return acc, y_pred