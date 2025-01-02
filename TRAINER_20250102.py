import os
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential

# Path to your data directory
data_dir = "C:\\Users\\223162D\\PycharmProjects\\year3\\movement_datasets"
n_time_steps = 60  # Number of timesteps for each sequence
step_size = 5     # Step size for the moving window
n_features = 36    # Number of features (x, y, z, visibility for multiple landmarks)

# Function to load and preprocess data using moving window
def load_exercise_data(folder, window_size, step_size):
    X = []
    y = []

    exercise_labels = os.listdir(folder)
    n_classes = len(exercise_labels)

    for label, exercise in enumerate(exercise_labels):
        exercise_folder = os.path.join(folder, exercise)

        for file in os.listdir(exercise_folder):
            file_path = os.path.join(exercise_folder, file)
            if file_path.lower().endswith('.csv'):
                data = pd.read_csv(file_path)
                sequence = data.values  # Convert DataFrame to numpy array

                # Generate windows using the specified window size and step size
                for start in range(0, len(sequence) - window_size + 1, step_size):
                    end = start + window_size
                    window = sequence[start:end]
                    if len(window) == window_size:
                        # Ensure the window is reshaped to (60, 33) for LSTM
                        if window.shape[1] == 36:  # Adjust according to your data
                            X.append(window)  # Directly append the window
                        y.append(label)

    return np.array(X), to_categorical(np.array(y), num_classes=n_classes), n_classes


# Load the data
X, y, n_classes = load_exercise_data(data_dir, n_time_steps, step_size)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Reshape input for LSTM (batch_size, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], n_time_steps, n_features))
X_test = X_test.reshape((X_test.shape[0], n_time_steps, n_features))

# Building the LSTM model
model = Sequential([
    Input(shape=(n_time_steps, n_features)),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=n_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=131, batch_size=128, validation_data=(X_test, y_test))

# Save the model
model.save("first.h5")
