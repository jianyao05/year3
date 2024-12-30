import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector():

    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, model_complexity=1, smooth_landmarks=self.smooth,
                                     enable_segmentation=False, smooth_segmentation=True,
                                     min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        # ------------------- STORES LANDMARK AND COORDINATES IN LIST eg. [id, x, y, z] --------------
        self.list = []


        # ------------------- COUNTER --------------
        self.SQUAT_COUNT = 0
        self.IMPROPER_SQUAT = 0

        self.INCORRECT_POSTURE = False

    # ------------------- FINDS AND DRAW LANDMARKS --------------
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


    # ------------------- CALCULATE ANGLES eg. [BELOW SHOULDER, SHOULDER, ELBOW] --------------
    def calculate_angle(self, point1, point2, point3):
        # Calculate vectors
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)

        # Dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Angle in radians
        angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

        # Convert to degrees
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    # ------------------- GET STATE BASED OFF ANGLES AND Z COORDINATES AND X? --------------
    def get_state(self, angle):
        state = None

        if 0 <= angle <= 30:
            state = 1
        elif 35 <= angle <= 75:
            state = 2
        elif 80 <= angle <= 100:
            state = 3
        return f"s{state}" if state else None

    def update_state_sequence(self, state):
        if state == 's2':
            if (('s3' not in self.list) and (self.list.count('s2')) == 0) or (('s3' in self.list) and (self.list.count('s2') == 1)):
                self.list.append(state)
                # If 's3' hasn’t been added yet, only one 's2' can be added.
                # If 's3' has been added, one more 's2' can be added, but only if it has appeared once before.
        elif state == 's3':
            if (state not in self.list) and ('s2' in self.list):
                self.list.append(state)
        return self.list