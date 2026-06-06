
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
    """
    Construiește un model CNN flexibil, cu ramuri dinamice generate în funcție de
    vectorii de intrare transmiși ca perechi nume_ramura=shape.

    Exemplu de apel:
        model = build_multi_branch_cnn_model(
            num_classes=3,
            hrv=(15, 1),
            eda=(6, 1)
        )
    """
    inputs = []
    branches = []

    # Iterăm dinamic prin toate intrările primite în kwargs (ex: hrv=(15,1), eda=(6,1))
    for branch_name, branch_shape in kwargs.items():
        if branch_shape is None:
            continue

        # 1. Definirea intrării unice pentru această ramură
        input_layer = Input(shape=branch_shape, name=f"input_{branch_name}")
        inputs.append(input_layer)

        # 2. Straturile convoluționale dedicate ramurii curente
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)

        # Pooling adaptiv (evităm erorile dacă dimensiunea temporală/feature este prea mică)
        pool_size = 2 if branch_shape[0] >= 2 else 1
        x = MaxPooling1D(pool_size=pool_size)(x)

        x = Dropout(0.2)(x)
        x = Flatten()(x)

        branches.append(x)

    # Validare de siguranță: dacă nu s-a transmis nicio ramură validă
    if not branches:
        raise ValueError("[ERROR] Trebuie să specifici cel puțin o intrare validă pentru a construi modelul!")

    # 3. Fuziunea Multimodală (Concatenare)
    # Dacă avem o singură ramură activă, nu mai este nevoie de concatenare
    if len(branches) > 1:
        fused = concatenate(branches, name="multimodal_fusion")
    else:
        fused = branches[0]

    # 4. Straturile dense comune de clasificare
    x = Dense(64, activation='relu')(fused)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax', name="output_stress_level")(x)

    # Crearea modelului final cu lista dinamică de intrări
    model = Model(inputs=inputs, outputs=outputs, name="Dynamic_Multi_Branch_CNN")

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model