from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from imblearn.metrics import geometric_mean_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import wilcoxon

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
    adam_optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
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
        epochs=200,
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

    return accuracy, mcc, f1, gm


def stopSignal(fold, X_train, X_val, X_test, y_train, y_val, y_test, padding_type):
    # Step 2: Data Preprocessing
    stop_character = -1
    if (padding_type == 'post'):
        X_train_stop = [seq + [stop_character] for seq in X_train]
        X_val_stop = [seq + [stop_character] for seq in X_val]
        X_test_stop = [seq + [stop_character] for seq in X_test]
    elif (padding_type == 'pre'):
        X_train_stop = [[stop_character] + seq for seq in X_train]
        X_val_stop = [[stop_character] + seq for seq in X_val]
        X_test_stop = [[stop_character] + seq for seq in X_test]

    # Step 2: Data Preprocessing
    max_length = max(len(seq) for seq in X_test)
    # print(max_length)

    # Perform padding on the sequences
    X_train_padded = pad_sequences(X_train_stop, maxlen=max_length, padding=padding_type, value=0.0)
    X_val_padded = pad_sequences(X_val_stop, maxlen=max_length, padding=padding_type, value=0.0)
    X_test_padded = pad_sequences(X_test_stop, maxlen=max_length, padding=padding_type, value=0.0)

    # ??? Reshape?
    X_test_padded = np.array(X_test_padded)
    X_val_padded = np.array(X_val_padded)
    X_train_padded = np.array(X_train_padded)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    model_input = Input(shape=(X_train_padded.shape[1], 1))
    x = Conv1D(32, 4, padding='same', kernel_initializer='he_normal', name="conv1d_1")(model_input)
    x = Conv1D(64, 4, padding='same', kernel_initializer='he_normal', name="conv1d_2")(x)
    x = Bidirectional(LSTM(256, unroll=True, name="bi_lstm"))(x)
    x = Dropout(0.2, name="dropout")(x)
    x = Dense(1, activation='sigmoid', name="output_dense")(x)
    model = Model(inputs=model_input, outputs=x)
    
    adam_optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
    
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.compile(loss="binary_crossentropy", optimizer=adam_optimizer)
    
    model.fit(
        X_train_padded,
        y_train,
        validation_data=(X_val_padded, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping_callback],
    )

    # Evaluate the model
    y_pred = model.predict(X_test_padded)
    y_pred_binary = (y_pred > 0.5).astype(int) # Turn the predictions into 1s and 0s ? Why this crudely

    # Metrics
    accuracy = accuracy_score(y_test, y_pred_binary, normalize=True)
    mcc = matthews_corrcoef(y_test, y_pred_binary, sample_weight=None)
    f1 = f1_score(y_test, y_pred_binary, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
    gm = geometric_mean_score(y_test, y_pred_binary.ravel(), labels=None, pos_label=1, average='multiclass', sample_weight=None, correction=0.0)

    with open((f'stop-signal-metrics-{padding_type}-{fold}.txt'), "a") as file:
        file.write(f"\nacc: {accuracy}\nmcc: {mcc}\nf1: {f1}\ngm: {gm}")

    return accuracy, mcc, f1, gm


## legit main
# 1. Create Train, test and validation data sets for both protein sequence data and their labels - 10 fold cross-validation
xDataSet, yDataSet = createDataSets()
no_stop_signal_post_acc = []
no_stop_signal_pre_acc = []
stop_signal_post_acc = []
stop_signal_pre_acc = []
# xDataSet and yDataSet are of same fold lenght

for i in range(len(xDataSet)):
    print("FOLD ", i)
    X_train = xDataSet[i].getTrainingDataSet()
    X_val = xDataSet[i].getValidationDataSet()
    X_test = xDataSet[i].getTestDataSet()
    y_train = yDataSet[i].getTrainingDataSet()
    y_val = yDataSet[i].getValidationDataSet()
    y_test = yDataSet[i].getTestDataSet()

    print("POST obican ")
    no_stop_signal_post_acc.append(regularPadding(i, X_train, X_val, X_test, y_train, y_val, y_test, 'post')[0])
    print("PRE obican ")
    no_stop_signal_pre_acc.append(regularPadding(i, X_train, X_val, X_test, y_train, y_val, y_test, 'pre')[0])
    print("POST  stop signal ")
    stop_signal_post_acc.append(stopSignal(i, X_train, X_val, X_test, y_train, y_val, y_test, 'post')[0])
    print("PRE  stop signal ")
    stop_signal_pre_acc.append(stopSignal(i, X_train, X_val, X_test, y_train, y_val, y_test, 'pre')[0])

stat_post, p_value_post = wilcoxon(no_stop_signal_post_acc, stop_signal_post_acc)
print("p-value (no-stop-signal vs. stop-signal, post-padding):", p_value_post)
print("stat - ", stat_post)

# "no-stop-signal" vs. "stop-signal" with "pre" padding
stat_pre, p_value_pre = wilcoxon(no_stop_signal_pre_acc, stop_signal_pre_acc)
print("p-value (no-stop-signal vs. stop-signal, pre-padding):", p_value_pre)
print("stat - ", stat_pre)

# pre vs post no stop signal
stat_no_stop, p_value_no_stop = wilcoxon(no_stop_signal_post_acc, no_stop_signal_pre_acc)
print("p-value (no-stop-signal post vs. no stop pre):", p_value_no_stop)
print("stat - ", stat_no_stop)

# pre vs post  stop signal
stat_stop, p_value_stop = wilcoxon(stop_signal_post_acc, stop_signal_pre_acc)
print("p-value (stop-signal post vs. stop-signal  pre-padding):", p_value_stop)
print("stat - ", stat_stop)

with open((f'results.txt'), "a") as file:
    file.write(f"\nno_stop_post: {no_stop_signal_post_acc}\nstop_post: {stop_signal_post_acc}\nno_stop_pre: {no_stop_signal_pre_acc}\nstop_pre: {stop_signal_pre_acc} \n")
    file.write(f"p-value (no-stop-signal vs. stop-signal, post-padding): {p_value_post} stats: {stat_post} \n")
    file.write(f"p-value (no-stop-signal vs. stop-signal, pre-padding): {p_value_pre} stats: {stat_pre} \n")
    file.write(f"p-value (no stop post vs pre): {p_value_no_stop} stats: {stat_no_stop}\n")
    file.write(f"p-value ( stop post vs pre): {p_value_stop} stats: {stat_stop}\n")
