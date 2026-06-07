from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, concatenate
from keras.optimizers import Adam


def build_lstm_model(input_shape, num_classes):

    inputs = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = LSTM(32, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_multi_branch_lstm_model(num_classes, branch_shapes):
    inputs = []
    branches = []

    for i, shape in enumerate(branch_shapes):
        input_layer = Input(shape=shape, name=f"input_lstm_branch_{i}")
        inputs.append(input_layer)

        x = LSTM(64, return_sequences=True, name=f"lstm1_branch_{i}")(input_layer)
        x = BatchNormalization(name=f"bn1_branch_{i}")(x)
        x = Dropout(0.2, name=f"drop1_branch_{i}")(x)

        x = LSTM(32, return_sequences=False, name=f"lstm2_branch_{i}")(x)
        x = BatchNormalization(name=f"bn2_branch_{i}")(x)
        x = Dropout(0.2, name=f"drop2_branch_{i}")(x)

        branches.append(x)

    #Fuziune
    if len(branches) > 1:
        fused = concatenate(branches, name="lstm_multimodal_fusion")
    else:
        fused = branches[0]

    # Straturi dense
    x = Dense(64, activation='relu', name="dense_lstm_fusion_1")(fused)
    x = Dropout(0.3, name="drop_lstm_fusion_1")(x)
    x = Dense(32, activation='relu', name="dense_lstm_fusion_2")(x)

    outputs = Dense(num_classes, activation='softmax', name="output_stress_level_lstm")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Multi_Branch_LSTM")

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model