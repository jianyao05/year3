import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import time
import math

# Initialization and loading model
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Load the trained LSTM model
model = tf.keras.models.load_model("THIRD_45stepsize.h5")

n_time_steps = 60                                           # Window size
step_size = 45                                              # Lower step size for more frequent predictions

# Labels and timing variables for movement detection
exercise_labels = ["ARMPIT RIGHT", "CIRCLE LEFT", "CIRCLE RIGHT", "CB LEFT", "CB RIGHT", "PENDULUM LEFT", "PENDULUM RIGHT", "FLEXION LEFT", "FLEXION RIGHT"]
lm_list = []
label = "Warmup...."
lm_count = 0


# Indices for leg landmarks in MediaPipe (leg movement detection)
arm_landmarks = {
    # RIGHT
    'right_shoulder': 12,           # FOR NORMALISATION
    'right_elbow': 14,
    'right_wrist': 16,
    'right_pinky': 18,
    'right_index': 20,
    'right_thumb': 22,
    # LEFT
    'left_shoulder': 11,            # FOR NORMALISATION
    'left_elbow': 13,
    'left_wrist': 15,
    'left_pinky': 17,
    'left_index': 19,
    'left_thumb': 21
}

# ========================================================================
def normalize_landmarks(landmarks):
    right_shoulder_x = landmarks[arm_landmarks['right_shoulder']].x
    right_shoulder_y = landmarks[arm_landmarks['right_shoulder']].y
    right_shoulder_z = landmarks[arm_landmarks['right_shoulder']].z
    left_shoulder_x = landmarks[arm_landmarks['left_shoulder']].x
    left_shoulder_y = landmarks[arm_landmarks['left_shoulder']].y
    left_shoulder_z = landmarks[arm_landmarks['right_shoulder']].z

    normalized_lm = []

    # Normalize right arm landmarks
    normalized_lm.extend([right_shoulder_x, right_shoulder_y, right_shoulder_z])
    for lm_id in ['right_elbow', 'right_wrist', 'right_pinky', 'right_index', 'right_thumb']:
        landmark = landmarks[arm_landmarks[lm_id]]
        normalized_x = landmark.x - right_shoulder_x
        normalized_y = landmark.y - right_shoulder_y
        normalized_z = landmark.z - right_shoulder_z
        normalized_lm.extend([normalized_x, normalized_y, normalized_z])

    # Normalize left arm landmarks
    normalized_lm.extend([left_shoulder_x, left_shoulder_y, left_shoulder_z])
    for lm_id in ['left_elbow', 'left_wrist', 'left_pinky', 'left_index', 'left_thumb']:
        landmark = landmarks[arm_landmarks[lm_id]]
        normalized_x = landmark.x - left_shoulder_x
        normalized_y = landmark.y - left_shoulder_y
        normalized_z = landmark.z - left_shoulder_z
        normalized_lm.extend([normalized_x, normalized_y, normalized_z])

    return normalized_lm



def make_landmark_timestep(results):
    """Extract and normalize leg landmarks, including ankle distance."""
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    # Normalize leg landmarks relative to hips
    normalized_landmarks = normalize_landmarks(landmarks)

    return normalized_landmarks


def draw_class_on_image(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_text = f"{label}"
    cv2.putText(img, status_text, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img
# ==========================================================================
def detect(model, lm_list):
    global label, lm_count

    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    predicted_class = np.argmax(results)
    label = exercise_labels[predicted_class]

    # Logic to manage detection outcomes and series of movements


warmup_frames = 60
i = 0

cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i += 1

    img = draw_class_on_image(img)

    if i > warmup_frames:
        c_lm = make_landmark_timestep(results)
        if c_lm:
            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = lm_list[step_size:]


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()