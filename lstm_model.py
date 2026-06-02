from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, concatenate
from keras.optimizers import Adam


def build_lstm_model(input_shape, num_classes):
    """
    Construiește o rețea Stacked LSTM pentru clasificarea stărilor de stres.
    """
    inputs = Input(shape=input_shape)

    # Primul strat LSTM - return_sequences=True pentru a transmite datele către următorul strat LSTM
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Al doilea strat LSTM - return_sequences=False pentru a trece la straturile Dense
    x = LSTM(32, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Strat Complet Conectat (Dense) pentru interpretarea feature-urilor extrase
    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_multi_branch_lstm_model(num_classes, branch_shapes):
    """
    Construiește un model LSTM multimodal flexibil pe bază de listă de shapes.
    Exemplu branch_shapes: [(15, 1), (6, 1)] pentru HRV și EDA.
    """
    inputs = []
    branches = []

    for i, shape in enumerate(branch_shapes):
        # 1. Strat de intrare anonim bazat pe index
        input_layer = Input(shape=shape, name=f"input_lstm_branch_{i}")
        inputs.append(input_layer)

        # 2. Ramura LSTM dedicată semnalului i
        x = LSTM(64, return_sequences=True, name=f"lstm1_branch_{i}")(input_layer)
        x = BatchNormalization(name=f"bn1_branch_{i}")(x)
        x = Dropout(0.2, name=f"drop1_branch_{i}")(x)

        x = LSTM(32, return_sequences=False, name=f"lstm2_branch_{i}")(x)
        x = BatchNormalization(name=f"bn2_branch_{i}")(x)
        x = Dropout(0.2, name=f"drop2_branch_{i}")(x)

        branches.append(x)

    # 3. Fuziune timpurie prin concatenare
    if len(branches) > 1:
        fused = concatenate(branches, name="lstm_multimodal_fusion")
    else:
        fused = branches[0]

    # 4. Straturi dense comune de clasificare
    x = Dense(64, activation='relu', name="dense_lstm_fusion_1")(fused)
    x = Dropout(0.3, name="drop_lstm_fusion_1")(x)
    x = Dense(32, activation='relu', name="dense_lstm_fusion_2")(x)

    outputs = Dense(num_classes, activation='softmax', name="output_stress_level_lstm")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Multi_Branch_LSTM")

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model