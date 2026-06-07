import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

# Importurile către arhitecturile tale
import cnn_model as cnn
import transformer_model as transformer
import lstm_model as lstm


def train_model(X_train, y_train, num_classes, model_name):

    print(f"\n[INFO] Starting training for {model_name} on {len(X_train)} samples...")
    BATCH_SIZE = 32


    if model_name == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print(f"[INFO] {model_name} training complete.")
        return model

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

    BATCH_SIZE = 32
    raw_preds = 0


    if model_name == 'RF':
        y_pred = model.predict(X_test)
        raw_preds = model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)


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

        branch_3d = branch_array.reshape(num_samples, num_features, 1)
        inputs_list.append(branch_3d)

        shapes_dict[f'branch_{b_idx}'] = (num_features, 1)
        print(f"     * Branch {b_idx} shape: {(num_features, 1)}")

    y_train_actual = np.array(y_train[0])
    y_train_cat = to_categorical(y_train_actual, num_classes=num_classes)

    input_names_dict = {f"input_branch_{i}": inputs_list[i] for i in range(num_branches)}

    train_dataset = tf.data.Dataset.from_tensor_slices((input_names_dict, y_train_cat))
    train_dataset = train_dataset.shuffle(1000, seed=42, reshuffle_each_iteration=True).batch(BATCH_SIZE,
                                                                                              drop_remainder=True)

    model = cnn.build_multi_branch_cnn_model(num_classes=num_classes, **shapes_dict)

    model.fit(train_dataset, epochs=20, verbose=1)
    print("[INFO] Multi-Branch CNN training complete.")

    return model


def predict_multi_branch_by_vector_count(model, X_test, y_test):

    if not X_test or len(X_test) == 0:
        raise ValueError("[ERROR] X_test este gol!")

    num_branches = len(X_test)
    num_samples = X_test[0].shape[0]
    BATCH_SIZE = 32

    inputs_list = []

    for b_idx in range(num_branches):
        df_branch = X_test[b_idx]
        branch_array = df_branch.values
        num_features = branch_array.shape[1]

        branch_3d = branch_array.reshape(num_samples, num_features, 1)
        inputs_list.append(branch_3d)

    probs = model.predict(inputs_list, batch_size=BATCH_SIZE, verbose=0)


    if isinstance(y_test, list) and len(y_test) > 0 and hasattr(y_test[0], '__len__') and len(y_test[0]) == num_samples:
        y_test_arr = np.array(y_test[0])

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

    print(f"\n[INFO] Pregătire date Train pentru Multi-Branch LSTM...")

    y_train = y_list[0]
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    y_train_cat = to_categorical(y_train, num_classes=num_classes)

    prepared_inputs = []
    branch_shapes = []

    for i, X_branch in enumerate(X_list):
        if hasattr(X_branch, 'values'):
            X_branch = X_branch.values

        X_3d = X_branch.reshape(X_branch.shape[0], X_branch.shape[1], 1)
        prepared_inputs.append(X_3d)

        branch_shapes.append((X_branch.shape[1], 1))
        print(f"  -> Ramura {i} procesată cu dimensiunea de input: {branch_shapes[-1]}")

    model = lstm.build_multi_branch_lstm_model(num_classes, branch_shapes)

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

    prepared_inputs = []
    for X_branch in X_list:
        if hasattr(X_branch, 'values'):
            X_branch = X_branch.values
        X_3d = X_branch.reshape(X_branch.shape[0], X_branch.shape[1], 1)
        prepared_inputs.append(X_3d)

    probs = model.predict(prepared_inputs, verbose=0)

    if len(probs) > 0:
        y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = np.zeros(len(X_list[0]))

    y_test = y_list[0]
    if hasattr(y_test, 'values'):
        y_test = y_test.values

    acc = accuracy_score(y_test, y_pred)

    return acc, y_pred, probs



def train_multi_rf_independent_branches(X_list, y_list, num_classes=3, n_estimators=100, class_weights_dict=None):

    print(f"\n[INFO] Pregătire date Train pentru Multi-RF (Independent pe ramuri)...")

    y_train = y_list[0]
    if hasattr(y_train, 'values'):
        y_train = y_train.values

    cw = class_weights_dict if class_weights_dict is not None else 'balanced'

    trained_rf_models = []

    for i, X_branch in enumerate(X_list):
        if hasattr(X_branch, 'values'):
            X_branch = X_branch.values

        print(f"  -> Antrenare RF independent pentru ramura {i} (Dimensiune: {X_branch.shape})")

        model = RandomForestClassifier(n_estimators=n_estimators, class_weight=cw, random_state=42)
        model.fit(X_branch, y_train)

        trained_rf_models.append(model)

    return trained_rf_models


def predict_multi_rf_independent_branches(models_list, X_list, y_list):

    y_test = y_list[0]
    if hasattr(y_test, 'values'):
        y_test = y_test.values

    num_samples = X_list[0].shape[0] if not hasattr(X_list[0], 'values') else X_list[0].values.shape[0]
    num_classes = len(models_list[0].classes_)

    summed_probs = np.zeros((num_samples, num_classes))

    for model, X_branch in zip(models_list, X_list):
        if hasattr(X_branch, 'values'):
            X_branch = X_branch.values

        probs = model.predict_proba(X_branch)

        summed_probs += probs

    avg_probs = summed_probs / len(models_list)

    y_pred = np.argmax(avg_probs, axis=1)

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
