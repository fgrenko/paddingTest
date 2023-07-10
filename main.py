from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


def regularPadding(X_train, X_test, y_train, y_test):

    # Step 2: Data Preprocessing
    max_length = max(len(seq) for seq in X_test)

    # Perform padding on the sequences  
    X_train_padded = pad_sequences(X_train, maxlen=max_length, padding='post', value=0.0)
    X_test_padded = pad_sequences(X_test, maxlen=max_length, padding='post', value=0.0)

    X_test_padded = np.array(X_test_padded)
    X_train_padded = np.array(X_train_padded)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Step 3: Model Training and Evaluation
    # Build and train the model
    model = Sequential()
    model.add(
        Dense(64, activation='relu', input_dim=max_length))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_padded, y_train, epochs=10, batch_size=32, shuffle=True)

    # Evaluate the model
    y_pred = model.predict(X_test_padded)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary, normalize=True)

    with open("accuracy.txt", "a") as file:
        file.write(f"{accuracy}\n")



def stopSignal(X_train, X_test, y_train, y_test):
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
