
import cnn_model as cnn
import transformer_model as transformer

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight



import tensorflow as tf
from keras.utils import to_categorical


def train_all_models_once(X_train, y_train, num_classes):
    BATCH_SIZE = 32
    print(f"\n[INFO] Starting training on {len(X_train)} samples...")

    # Random Forest
    print("  -> Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Class weights
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(weights))

    print(f"  Class Weights: {class_weights_dict}")

    # date
    X_train_dl = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    y_train_cat = to_categorical(y_train, num_classes=num_classes)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_dl, y_train_cat))
    # drop_remainder=True la antrenare pentru stabilitate
    train_dataset = train_dataset.shuffle(1000, seed=42, reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)


    # CNN
    print("  -> Training CNN...")
    cnn_model = cnn.build_cnn_model((X_train.shape[1], 1), num_classes)
    cnn_model.fit(train_dataset, epochs=20, verbose=0)
    #cnn_model.fit(train_dataset, epochs=20, verbose=0, class_weight=class_weights_dict)


    # Transformer
    print("  -> Training Transformer...")
    trans_model = transformer.build_transformer_model((X_train.shape[1], 1), num_classes)
    trans_model.fit(train_dataset, epochs=20, verbose=0)
    #trans_model.fit(train_dataset, epochs=20, verbose=0, class_weight=class_weights_dict)

    print("[INFO] Training complete.")
    return rf_model, cnn_model, trans_model

def predict_on_test_data(models, X_test, y_test):

    rf_model, cnn_model, trans_model = models
    BATCH_SIZE = 32

    # Random Forest
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # Data prep DL
    X_test_dl = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    test_dataset = tf.data.Dataset.from_tensor_slices(X_test_dl)
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)

    # CNN
    probs_cnn = cnn_model.predict(test_dataset, verbose=0)
    if len(probs_cnn) > 0:
        y_pred_cnn = np.argmax(probs_cnn, axis=1)
        acc_cnn = accuracy_score(y_test, y_pred_cnn)
    else:
        y_pred_cnn = np.zeros_like(y_test)  # Fallback
        acc_cnn = 0.0

    # Transformer
    probs_trans = trans_model.predict(test_dataset, verbose=0)
    if len(probs_trans) > 0:
        y_pred_trans = np.argmax(probs_trans, axis=1)
        acc_trans = accuracy_score(y_test, y_pred_trans)
    else:
        y_pred_trans = np.zeros_like(y_test)
        acc_trans = 0.0

    return {
        'acc_rf': acc_rf,
        'acc_cnn': acc_cnn,
        'acc_transformer': acc_trans,
        'y_pred_rf': y_pred_rf,
        'y_pred_cnn': y_pred_cnn,
        'y_pred_trans': y_pred_trans
    }