from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

# Step 1: Prepare the Dataset
df = pd.read_csv('amp.csv')

# Assuming the sequences are stored in a column named 'sequence' in the CSV file
sequences = df['sequence'].tolist()
labels = df['label'].tolist()

amino_acids = list(set(''.join(sequences)))
amino_to_num = {amino: num for num, amino in enumerate(amino_acids)}
numerical_sequences = [[amino_to_num[amino] for amino in sequence] for sequence in sequences]

# Assuming you have a list of amino acid sequences called 'sequences' and a corresponding list of labels called 'labels'
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(numerical_sequences, labels, test_size=0.2, random_state=42)

# Step 2: Data Preprocessing
# Add stop signal character at the end of each sequence
# X_train_stop = [seq + ['STOP'] for seq in X_train]
# X_test_stop = [seq + ['STOP'] for seq in X_test]

# Determine the maximum length of the sequences (including the stop signal)
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
model.add(Dense(64, activation='relu', input_dim=max_length))  # Adjust the model architecture as per your requirements
model.add(Dense(1, activation='sigmoid'))  # Assuming it's a binary classification problem

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_padded, y_train, epochs=10, batch_size=32, shuffle=True)

# Evaluate the model
y_pred = model.predict(X_test_padded)
y_pred_binary = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary, normalize=True)

print("Accuracy:", accuracy)
