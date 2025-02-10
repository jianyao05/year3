import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

# Load the model
model_path = "fyp_demo.h5"  # Replace with your model path
model = load_model(model_path)

# Load and preprocess the test data
# Replace 'test_data_dir' with the directory containing your test CSV files
import os
import pandas as pd

test_data_dir = ("C:\\Users\\223162D\\PycharmProjects\\year3\\test"
                 "_datasets")  # Replace with your test dataset folder
n_time_steps = 60
n_features = 36


def load_test_data(folder):
    X = []
    y = []
    labels = sorted(os.listdir(folder))  # Ensure consistent label ordering
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label, idx in label_map.items():
        exercise_folder = os.path.join(folder, label)
        for file in os.listdir(exercise_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(exercise_folder, file)
                data = pd.read_csv(file_path).values  # Convert to numpy array
                if data.shape[0] >= n_time_steps:  # Ensure sufficient timesteps
                    data = data[:n_time_steps]  # Trim or pad to n_time_steps
                    X.append(data)
                    y.append(idx)

    X = np.array(X)
    y = np.array(y)
    return X, y, label_map


# Load test data
X_test, y_test_labels, label_map = load_test_data(test_data_dir)

# One-hot encode y_test_labels
encoder = OneHotEncoder(sparse_output=False)
y_test = encoder.fit_transform(y_test_labels.reshape(-1, 1))

# Ensure the test data shape is correct
X_test = X_test.reshape((X_test.shape[0], n_time_steps, n_features))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions
y_pred_probs = model.predict(X_test)  # Predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted class labels
y_true = np.argmax(y_test, axis=1)  # True class labels

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
report = classification_report(y_true, y_pred, target_names=list(label_map.keys()))
print("Classification Report:")
print(report)

# Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
