
from keras.models import Model
from keras.layers import Dense, Conv1D, Dropout, Input, MultiHeadAttention, LayerNormalization, \
    Add, GlobalAveragePooling1D
from keras.optimizers import Adam



def build_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    attention = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    attention = Dropout(0.1)(attention)
    res = Add()([inputs, attention])
    x = LayerNormalization(epsilon=1e-6)(res)

    x_ff = Conv1D(filters=32, kernel_size=1, activation="relu")(x)
    x_ff = Dropout(0.1)(x_ff)
    x_ff = Conv1D(filters=1, kernel_size=1)(x_ff)
    res = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(res)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=0.0001)  # Default e 0.001, incercam de 10x mai mic
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
