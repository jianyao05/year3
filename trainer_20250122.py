import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
DATA_DIR = "C:\\Users\\223162D\\PycharmProjects\\year3\\movement_datasets"
MODEL_SAVE_PATH = "NEW_CODE_V4.h5"
N_TIME_STEPS = 60  # Number of timesteps per sequence
STEP_SIZE = 10  # Step size for moving window
N_FEATURES = 36  # Number of features (x, y, z, visibility)
EPOCHS = 131  # Number of epochs for training
BATCH_SIZE = 128  # Batch size for training
VALIDATION_SPLIT = 0.2  # Validation split for training


# Function to load and preprocess data
def load_exercise_data(folder, window_size, step_size):
    X, y = [], []
    exercise_labels = os.listdir(folder)
    n_classes = len(exercise_labels)

    for label, exercise in enumerate(exercise_labels):
        exercise_folder = os.path.join(folder, exercise)

        for file in os.listdir(exercise_folder):
            if file.lower().endswith('.csv'):
                file_path = os.path.join(exercise_folder, file)
                data = pd.read_csv(file_path).values  # Convert to numpy array

                # Create sliding windows
                for start in range(0, len(data) - window_size + 1, step_size):
                    end = start + window_size
                    window = data[start:end]
                    if window.shape[0] == window_size and window.shape[1] == N_FEATURES:
                        X.append(window)
                        y.append(label)

    return np.array(X), to_categorical(np.array(y), num_classes=n_classes), n_classes


# Load the dataset
print("Loading data...")
X, y, n_classes = load_exercise_data(DATA_DIR, N_TIME_STEPS, STEP_SIZE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Build the LSTM model
def build_model(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=n_classes, activation="softmax")
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


print("Building model...")
model = build_model((N_TIME_STEPS, N_FEATURES), n_classes)

# Train the model
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=1
)

# Save the trained model
print(f"Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model training complete and saved.")
