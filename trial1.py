import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import math

"""
model path: refers to the model being used
"""

class FrozenShoulder:
    def __init__(self, model_path, n_time_steps = 60, step_size = 45):
        # Initialize MediaPipe
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

        # Initialise Model & the thresholds and values
        self.model = tf.keras.models.load_model(model_path)
        self.n_time_steps = n_time_steps
        self.step_size = step_size

        # Labels for Indices of leg landmarks in MediaPipe (leg movement detection)
        self.arm_landmarks = {
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
        
        # Labels for exercises
        self.exercise_labels = [
            "ARMPIT RIGHT"  ,    
            "CIRCLE LEFT"   ,    "CIRCLE RIGHT", 
            "CB LEFT"       ,    "CB RIGHT",
            "PENDULUM LEFT" ,    "PENDULUM RIGHT", 
            "FLEXION LEFT"  ,    "FLEXION RIGHT"
        ]
        self.label = "Warmup...."
        self.lm_list = []  # Store landmark data for prediction

        # Exercise state tracking
        self.state_sequence = []
        self.count = 0
        self.improper_count = 0 ### may not be needed
        pass

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    # ------------------- STORES LANDMARK AND COORDINATES IN LIST eg. [id, x, y, z] --------------
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(h, w)
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def normalize_landmarks(self, landmarks):
        right_shoulder_x = landmarks[self.arm_landmarks['right_shoulder']].x
        right_shoulder_y = landmarks[self.arm_landmarks['right_shoulder']].y
        right_shoulder_z = landmarks[self.arm_landmarks['right_shoulder']].z
        left_shoulder_x = landmarks[self.arm_landmarks['left_shoulder']].x
        left_shoulder_y = landmarks[self.arm_landmarks['left_shoulder']].y
        left_shoulder_z = landmarks[self.arm_landmarks['left_shoulder']].z

        normalized_lm = []

        # Normalize right arm landmarks
        normalized_lm.extend([right_shoulder_x, right_shoulder_y, right_shoulder_z])
        for lm_id in ['right_elbow', 'right_wrist', 'right_pinky', 'right_index', 'right_thumb']:
            landmark = landmarks[self.arm_landmarks[lm_id]]
            normalized_x = landmark.x - right_shoulder_x
            normalized_y = landmark.y - right_shoulder_y
            normalized_z = landmark.z - right_shoulder_z
            normalized_lm.extend([normalized_x, normalized_y, normalized_z])

        # Normalize left arm landmarks
        normalized_lm.extend([left_shoulder_x, left_shoulder_y, left_shoulder_z])
        for lm_id in ['left_elbow', 'left_wrist', 'left_pinky', 'left_index', 'left_thumb']:
            landmark = landmarks[self.arm_landmarks[lm_id]]
            normalized_x = landmark.x - left_shoulder_x
            normalized_y = landmark.y - left_shoulder_y
            normalized_z = landmark.z - left_shoulder_z
            normalized_lm.extend([normalized_x, normalized_y, normalized_z])

        return normalized_lm

    def make_landmark_timestep(self, results):
        """Extract and normalize landmarks for a timestep."""
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        return self.normalize_landmarks(landmarks)
    
    def draw_class_on_image(self, img):
        """Display the predicted class label on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, self.label, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img
    
    # more things here

    def detect_movement(self, lm_list):
        """Detect movement and update label."""
        lm_list = np.expand_dims(np.array(lm_list), axis=0)
        results = self.model.predict(lm_list)
        self.label = self.exercise_labels[np.argmax(results)]

    def process_frame(self, img, results):
        """Process each frame, extract landmarks, and predict movement."""
        img = self.draw_class_on_image(img)
        c_lm = self.make_landmark_timestep(results)
        if c_lm:
            self.lm_list.append(c_lm)
            if len(self.lm_list) == self.n_time_steps:
                threading.Thread(target=self.detect_movement, args=(self.lm_list,)).start()
                self.lm_list = self.lm_list[self.step_size:]
        return img
    

    def angle(self):
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)







# Main script
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = FrozenShoulder("20250106v1_45stepsize.h5")

    warmup_frames = 60
    frame_count = 0

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.pose.process(imgRGB)
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        print
        frame_count += 1

        if frame_count > warmup_frames:
            img = detector.process_frame(img, results)

        cv2.imshow("Exercise Detector", img)
        print(lmList[2])
        print(lmList[2][1:])
        print(results.pose_landmarks.landmark[2])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()