from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from imblearn.metrics import geometric_mean_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Otović
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D
from tensorflow.keras.models import Model#, Sequential
import tensorflow.keras as keras

import numpy as np

from createDataSet import createDataSets

def regularPadding(fold, X_train, X_val, X_test, y_train, y_val, y_test, padding_type):
    # Step 2: Data Preprocessing
    max_length = max(len(seq) for seq in X_test)
    # print(max_length)

    # Perform padding on the sequences
    X_train_padded = pad_sequences(X_train, maxlen=max_length, padding=padding_type, value=0.0)
    X_val_padded = pad_sequences(X_val, maxlen=max_length, padding=padding_type, value=0.0)
    X_test_padded = pad_sequences(X_test, maxlen=max_length, padding=padding_type, value=0.0)

    # ??? Reshape?
    X_test_padded = np.array(X_test_padded)
    X_val_padded = np.array(X_val_padded)
    X_train_padded = np.array(X_train_padded)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    # Step 3: Model Training and Evaluation
    # Build and train the model

    # Definiranje modela. Za ovaj model smo mi proveli optimizaciju hiperparametara.
    # model_input = Input(shape=X_train_padded.shape)
    # Shape should be (len of sequence, how many sequences there are)
    # print(type(X_train_padded))
    # print(X_train_padded)
    # Issue - Otović tu dobije X_train u formatu (50, 10, 20) - as in, 50 puta array od 10 elemenata duljine 20
    # Mi dobijemo jedanput 7xxx/max_len elemenata duljine max_len
    # Ovaj shape nisan ni sigura kako točno dela so?????
    model_input = Input(shape=(X_train_padded.shape[1], 1))
    # print(len(X_train_padded), X_train_padded.shape[1])
    # print(model_input)
    # Same padding should have no effect since all data has already been padded beforehand
    x = Conv1D(32, 4, padding='same', kernel_initializer='he_normal', name="conv1d_1")(model_input)
    # print("conv1d 1 generated")
    x = Conv1D(64, 4, padding='same', kernel_initializer='he_normal', name="conv1d_2")(x)
    # print("conv1d 2 generated")
    x = Bidirectional(LSTM(256, unroll=True, name="bi_lstm"))(x)
    # print("bidir generated")
    x = Dropout(0.2, name="dropout")(x)
    # print("dropout generated")
    x = Dense(1, activation='sigmoid', name="output_dense")(x)
    # print("dense generated")
    model = Model(inputs=model_input, outputs=x)
    # print("model generated")

    # Trening modela. Za ovaj learning rate smo takoder utvrdili da je optimalan.
    adam_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    # print("adam generated")

    # Definicija early stoppinga koji ce automatski zaustaviti treniranje kada se loss na validacijskom setu prestane smanjivati.
    # Restore_best_weights=true omogucava da zadrzimo model koji je imao najmanji loss na validacijskom setu.
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # print("bidir generated")
    model.compile(loss="binary_crossentropy", optimizer=adam_optimizer)
    # print("compile generated")

    model.fit(
        X_train_padded,
        y_train,
        validation_data=(X_val_padded, y_val),
        epochs=3, # 200 odmah nazad nakon ča skužin da delaju metrike
        batch_size=32,
        callbacks=[early_stopping_callback],
    )
    # print("fit generated")
    # model = Sequential()
    # model.add(
    #     Dense(64, activation='relu', input_dim=max_length))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.fit(X_train_padded, y_train, epochs=10, batch_size=32, shuffle=True)

    # Evaluate the model
    y_pred = model.predict(X_test_padded)
    y_pred_binary = (y_pred > 0.5).astype(int) # Turn the predictions into 1s and 0s ? Why this crudely

    # Metrics
    accuracy = accuracy_score(y_test, y_pred_binary, normalize=True)
    # print("acc generated")
    mcc = matthews_corrcoef(y_test, y_pred_binary, sample_weight=None)
    # print("mcc generated")
    f1 = f1_score(y_test, y_pred_binary, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    # print("f1 generated")
    # DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    # y = column_or_1d(y, dtype=self.classes_.dtype, warn=True)
    gm = geometric_mean_score(y_test, y_pred_binary.ravel(), labels=None, pos_label=1, average='multiclass', sample_weight=None, correction=0.0)
    # print("gm generated")

    with open((f'no-stop-signal-metrics-{padding_type}-{fold}.txt'), "a") as file:
        file.write(f"\nacc: {accuracy}\nmcc: {mcc}\nf1: {f1}\ngm: {gm}")


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
for i in range(len(xDataSet)):
    X_train = xDataSet[i].getTrainingDataSet()
    X_val = xDataSet[i].getValidationDataSet()
    X_test = xDataSet[i].getTestDataSet()
    y_train = yDataSet[i].getTrainingDataSet()
    y_val = yDataSet[i].getValidationDataSet()
    y_test = yDataSet[i].getTestDataSet()

    regularPadding(i, X_train, X_val, X_test, y_train, y_val, y_test, 'post')
    regularPadding(i, X_train, X_val, X_test, y_train, y_val, y_test, 'pre')
