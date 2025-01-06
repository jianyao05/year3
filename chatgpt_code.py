import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import math


class ExerciseDetector:
    def __init__(self, model_path, n_time_steps=60, step_size=45):
        # Initialize MediaPipe and model
        self.pose = mp.solutions.pose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

        self.model = tf.keras.models.load_model(model_path)
        self.n_time_steps = n_time_steps
        self.step_size = step_size

        # Labels for exercises
        self.exercise_labels = [
            "ARMPIT RIGHT", "CIRCLE LEFT", "CIRCLE RIGHT", "CB LEFT", "CB RIGHT",
            "PENDULUM LEFT", "PENDULUM RIGHT", "FLEXION LEFT", "FLEXION RIGHT"
        ]
        self.label = "Warmup...."
        self.lm_list = []  # Store landmark data for prediction

        # Squat state tracking
        self.state_sequence = []
        self.count = 0
        self.improper_count = 0

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
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def normalize_landmarks(self, landmarks):
        """Normalize arm landmarks based on shoulder positions."""
        arm_landmarks = {
            'right_shoulder': 12, 'right_elbow': 14, 'right_wrist': 16, 'right_pinky': 18,
            'right_index': 20, 'right_thumb': 22, 'left_shoulder': 11, 'left_elbow': 13,
            'left_wrist': 15, 'left_pinky': 17, 'left_index': 19, 'left_thumb': 21
        }

        normalized_lm = []

        for side in ['right', 'left']:
            shoulder = arm_landmarks[f"{side}_shoulder"]
            sx, sy, sz = landmarks[shoulder].x, landmarks[shoulder].y, landmarks[shoulder].z
            normalized_lm.extend([sx, sy, sz])

            for joint in ['elbow', 'wrist', 'pinky', 'index', 'thumb']:
                lm = landmarks[arm_landmarks[f"{side}_{joint}"]]
                normalized_x, normalized_y, normalized_z = lm.x - sx, lm.y - sy, lm.z - sz
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

    def calculate_angle(self, point1, point2, point3):
        """Calculate the angle between three points."""
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
        return np.degrees(angle_radians)

    def update_state_sequence(self, state):
        """Update state sequence based on detected angles."""
        if state == 's2':
            if ('s3' not in self.state_sequence and self.state_sequence.count('s2') == 0) or (
                's3' in self.state_sequence and self.state_sequence.count('s2') == 1
            ):
                self.state_sequence.append(state)
        elif state == 's3' and 's2' in self.state_sequence:
            if state not in self.state_sequence:
                self.state_sequence.append(state)
        return self.state_sequence

    def count_reps(self, state):
        """Count proper and improper repetitions."""
        if state == 's1':
            if len(self.state_sequence) == 3:
                self.count += 1
            elif 's2' in self.state_sequence and len(self.state_sequence) == 1:
                self.improper_count += 1
            self.state_sequence = []  # Reset sequence

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


# Main script
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = ExerciseDetector("20250106v1_45stepsize.h5")

    warmup_frames = 60
    frame_count = 0

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.pose.process(imgRGB)
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        # print(lmList)
        if len(lmList) != 0:
            shoulder = lmList[11][1:]
            elbow = lmList[13][1:]
            point_below_shoulder = [shoulder[0], shoulder[1] + 0.2,
                                    shoulder[2]]  # Add 0.2 to y to move below the shoulder

            angle = detector.calculate_angle(point_below_shoulder, shoulder, elbow)
            print(angle)
        frame_count += 1

        if frame_count > warmup_frames:
            img = detector.process_frame(img, results)

        cv2.imshow("Exercise Detector", img)
        print(detector.count, detector.improper_count, detector.state_sequence)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
