from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
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