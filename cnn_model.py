
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, concatenate
from keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))

    pool_size = 2 if input_shape[0] >= 2 else 1
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model





def build_multi_branch_cnn_model(num_classes, **kwargs):

    inputs = []
    branches = []

    for branch_name, branch_shape in kwargs.items():
        if branch_shape is None:
            continue

        input_layer = Input(shape=branch_shape, name=f"input_{branch_name}")
        inputs.append(input_layer)

        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)

        pool_size = 2 if branch_shape[0] >= 2 else 1
        x = MaxPooling1D(pool_size=pool_size)(x)

        x = Dropout(0.2)(x)
        x = Flatten()(x)

        branches.append(x)

    if not branches:
        raise ValueError("[ERROR] Trebuie să specifici cel puțin o intrare validă pentru a construi modelul!")

    # Fuziunea Multimodală
    if len(branches) > 1:
        fused = concatenate(branches, name="multimodal_fusion")
    else:
        fused = branches[0]

    # Straturile dense comune
    x = Dense(64, activation='relu')(fused)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax', name="output_stress_level")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Dynamic_Multi_Branch_CNN")

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model