from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Otović
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D
from tensorflow.keras.models import Model#, Sequential
import tensorflow.keras as keras

import numpy as np

from createDataSet import createDataSets

def regularPadding(X_train, X_val, X_test, y_train, y_val, y_test, padding_type):
    # Step 2: Data Preprocessing
    max_length = max(len(seq) for seq in X_test)
    # print(max_length)

    # Perform padding on the sequences
    X_train_padded = pad_sequences(X_train, maxlen=max_length, padding=padding_type, value=0.0)
    X_val_padded = pad_sequences(X_val, maxlen=max_length, padding=padding_type, value=0.0)
    X_test_padded = pad_sequences(X_test, maxlen=max_length, padding=padding_type, value=0.0)

    # ??? Reshape?
    # X_test_padded = np.array(X_test_padded)
    # X_val_padded = np.array(X_val_padded)
    # X_train_padded = np.array(X_train_padded)
    # y_train = np.array(y_train)
    # y_val = np.array(y_val)
    # y_test = np.array(y_test)

    # Step 3: Model Training and Evaluation
    # Build and train the model

    # Definiranje modela. Za ovaj model smo mi proveli optimizaciju hiperparametara.
    # model_input = Input(shape=X_train_padded.shape)
    # Shape should be (len of sequence, how many sequences there are)
    print(type(X_train_padded))
    print(X_train_padded)
    # Issue - Otović tu dobije X_train u formatu (50, 10, 20) - as in, 50 puta array od 10 elemenata duljine 20
    # Mi dobijemo jedanput 7xxx/max_len elemenata duljine max_len
    # Ovaj shape nisan ni sigura kako točno dela so?????
    model_input = Input(X_train_padded.shape[1])
    # print(len(X_train_padded), X_train_padded.shape[1])
    print(model_input)
    exit()
    # Same padding should have no effect since all data has already been padded beforehand
    x = Conv1D(32, 4, padding='same', kernel_initializer='he_normal', name="conv1d_1")(model_input)
    print("conv1d 1 generated")
    x = Conv1D(64, 4, padding='same', kernel_initializer='he_normal', name="conv1d_2")(x)
    print("conv1d 2 generated")
    x = Bidirectional(LSTM(256, unroll=True, name="bi_lstm"))(x)
    print("bidir generated")
    x = Dropout(0.2, name="dropout")(x)
    print("dropout generated")
    x = Dense(1, activation='sigmoid', name="output_dense")(x)
    print("dense generated")
    model = Model(inputs=model_input, outputs=x)
    print("model generated")

    # Trening modela. Za ovaj learning rate smo takoder utvrdili da je optimalan.
    adam_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    print("adam generated")

    # Definicija early stoppinga koji ce automatski zaustaviti treniranje kada se loss na validacijskom setu prestane smanjivati.
    # Restore_best_weights=true omogucava da zadrzimo model koji je imao najmanji loss na validacijskom setu.
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    print("bidir generated")
    model.compile(loss="binary_crossentropy", optimizer=adam_optimizer)
    print("compile generated")

    print(X_train_padded)
    print(y_train)
    print(X_val_padded)
    print(y_val)
    exit()

    model.fit(
        X_train_padded,
        y_train,
        validation_data=(X_val_padded, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping_callback],
    )
    print("fit generated")
    # model = Sequential()
    # model.add(
    #     Dense(64, activation='relu', input_dim=max_length))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(X_train_padded, y_train, epochs=10, batch_size=32, shuffle=True)

    # Evaluate the model
    y_pred = model.predict(X_test_padded)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary, normalize=True)

    with open("accuracy.txt", "a") as file:
        file.write(f"{accuracy}\n")



def stopSignal(X_train, X_test, y_train, y_test, padding_type):
    # Step 2: Data Preprocessing
    stop_character = -1
    X_train_stop = [seq + [stop_character] for seq in X_train]
    X_test_stop = [seq + [stop_character] for seq in X_test]

    max_length = max(len(seq) for seq in X_test_stop)

    # Perform padding on the sequences
    X_train_padded = pad_sequences(X_train_stop, maxlen=max_length, padding='post', value=0)
    X_test_padded = pad_sequences(X_test_stop, maxlen=max_length, padding='post', value=0)

    X_test_padded = np.array(X_test_padded)
    X_train_padded = np.array(X_train_padded)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Step 3: Model Training and Evaluation
    model = Sequential()
    model.add(
        Dense(64, activation='relu', input_dim=max_length))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_padded, y_train, epochs=10, batch_size=32, shuffle=True)

    # Evaluate the model
    y_pred = model.predict(X_test_padded)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)

    with open("accuracy.txt", "a") as file:
        file.write(f"{accuracy}\n")


## legit main
# 1. Create Train, test and validation data sets for both protein sequence data and their labels - 10 fold cross-validation
xDataSet, yDataSet = createDataSets()

# xDataSet and yDataSet are of same fold lenght
# Siguro 10 folda
for i in range(10):
    X_train = xDataSet[i].getTrainingDataSet()
    X_val = xDataSet[i].getValidationDataSet()
    X_test = xDataSet[i].getTestDataSet()
    y_train = yDataSet[i].getTrainingDataSet()
    y_val = yDataSet[i].getValidationDataSet()
    y_test = yDataSet[i].getTestDataSet()

    regularPadding(X_train, X_val, X_test, y_train, y_val, y_test, 'post')
