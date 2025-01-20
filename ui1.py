import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import math
import streamlit as st
import tempfile
import time

DEMO_VIDEO = "demo_vid.mp4"


class FrozenShoulder:
    def __init__(self, model_path, n_time_steps=60, step_size=45):
        # Initialize MediaPipe
        self.ROM_left_flexion = 90
        self.ROM_right_flexion = 90
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
            'right_shoulder': 12,  # FOR NORMALISATION
            'right_elbow': 14,
            'right_wrist': 16,
            'right_pinky': 18,
            'right_index': 20,
            'right_thumb': 22,
            # LEFT
            'left_shoulder': 11,  # FOR NORMALISATION
            'left_elbow': 13,
            'left_wrist': 15,
            'left_pinky': 17,
            'left_index': 19,
            'left_thumb': 21
        }

        # Labels for exercises
        self.exercise_labels = [
            "ARMPIT RIGHT",
            "CIRCLE LEFT", "CIRCLE RIGHT",
            "CB LEFT", "CB RIGHT",
            "PENDULUM LEFT", "PENDULUM RIGHT",
            "FLEXION LEFT", "FLEXION RIGHT"
        ]
        self.label = "Warmup...."
        self.lm_list = []  # Store landmark data for prediction

        # Exercise state tracking
        self.list = []
        self.state_sequence = []
        self.LEFTFLEXION_COUNTER = 0
        self.RIGHTFLEXION_COUNTER = 0

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
                h, w, c = img.shape  # 480 640
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
        cv2.putText(img, "L-Flexion: {}".format(str(self.LEFTFLEXION_COUNTER)), (510, 440), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "R-Flexion: {}".format(str(self.RIGHTFLEXION_COUNTER)), (510, 470), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
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

    def angle(self, img, LS, LE, RS, RE):

        if self.label == "FLEXION LEFT":
            cv2.line(img, (LS[0], 400), LS[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.line(img, LS[:2], LE[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.circle(img, LS[:2], 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, LE[:2], 5, (255, 0, 0), cv2.FILLED)
            reference = LS[0], 1000, LS[2]
            vector1 = np.array(reference) - np.array(LS)
            vector2 = np.array(LE) - np.array(LS)
        elif self.label == "FLEXION RIGHT":
            cv2.line(img, (RS[0], 400), RS[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.line(img, RS[:2], RE[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.circle(img, RS[:2], 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, RE[:2], 5, (0, 0, 255), cv2.FILLED)
            reference = RS[0], 1000, RS[2]
            vector1 = np.array(reference) - np.array(RS)
            vector2 = np.array(RE) - np.array(RS)
        else:
            reference = RS[0], 1000, RS[2]
            vector1 = np.array(reference) - np.array(RS)
            vector2 = np.array(RE) - np.array(RS)

        # Dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Angle in radians
        angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

        # Convert to degrees
        angle_degrees = np.degrees(angle_radians)
        cv2.putText(img, str(int(angle_degrees)), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return angle_degrees

    def get_state(self, angle):
        state = None
        if self.label == "FLEXION LEFT":
            ROM = self.ROM_left_flexion
        elif self.label == "FLEXION RIGHT":
            ROM = self.ROM_right_flexion
        else:
            ROM = self.ROM_right_flexion
        if ROM - 100 <= angle <= ROM - 80:  # 0  < ANGLES < 20
            state = 1
        elif ROM - 50 <= angle <= ROM - 10:  # 30  < ANGLES < 80
            state = 2
        elif ROM <= angle <= ROM + 10:  # 90  < ANGLES < 100
            state = 3
            print(ROM)
            print(self.ROM_left_flexion)
        return f"s{state}" if state else None

    def update_state_sequence(self, state):
        if state == 's2':
            if (('s3' not in self.list) and (self.list.count('s2')) == 0) or (
                    ('s3' in self.list) and (self.list.count('s2') == 1)):
                self.list.append(state)
        elif state == 's3':
            if (state not in self.list) and ('s2' in self.list):
                self.list.append(state)
        return self.list

    def counter(self, img, state):
        if state == 's1':
            if len(self.list) == 3:
                if self.label == "FLEXION LEFT":
                    self.LEFTFLEXION_COUNTER += 1
                elif self.label == "FLEXION RIGHT":
                    self.RIGHTFLEXION_COUNTER += 1
                else:
                    pass
            elif 's2' in self.list and len(self.list) == 1:
                print("Improper Form")

            self.list = []

        return self.LEFTFLEXION_COUNTER, self.RIGHTFLEXION_COUNTER


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width / float(w)
        dim = int(w * r), height
    else:
        r = width / float(w)
        dim = width, int(h * r)

    # resizing of image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


if "threshold" not in st.session_state:
    st.session_state.ROM_left_flexion = 90  # Initialize threshold in session state
    st.session_state.ROM_right_flexion = 90

st.set_page_config(layout="wide")
st.title("Frozen Shoulder Rehabilitation Model")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>


    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Frozen Shoulder Sidebar")
st.sidebar.subheader("Parameters")

st.cache_resource()

app_mode = st.sidebar.selectbox("Choose the App Mode", ["Settings", "Video"])

if app_mode == "Settings":
    st.header("Settings")

    ###
    st.subheader("**SHOULDER FLEXION**")
    epi1, epi2, epi3, epi4 = st.columns(4)
    epi1.slider("**SHOULDER FLEXION LEFT [REPETITIONS PER SET]**", min_value=5, max_value=30, value=10)
    st.session_state.ROM_left_flexion = epi2.slider("**SHOULDER FLEXION LEFT [ANGLE THRESHOLD]**", min_value=70,
                                                    max_value=150, value=st.session_state.ROM_left_flexion)
    epi3.slider("**SHOULDER FLEXION RIGHT [REPETITIONS PER SET]**", min_value=5, max_value=30, value=10)
    st.session_state.ROM_right_flexion = epi4.slider("**SHOULDER FLEXION RIGHT [ANGLE THRESHOLD]**", min_value=70,
                                                     max_value=150, value=st.session_state.ROM_right_flexion)


elif app_mode == "Video":
    st.empty()
    c1, c2, c3 = st.columns([0.7, 0.15, 0.15], border=True)
    with c2:
        count_left_shoulder = c2.markdown("**SHOULDER LEFT: {}**".format(0))
        count_right_shoulder = c2.markdown("**SHOULDER RIGHT: {}**".format(0))

    with c3:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown('0')
        st.divider()
        st.markdown('**Range of Motion**')
        kpi2_text = st.markdown('0')
        st.divider()
        st.markdown('**Sets Completed**')
        kpi3_text = st.markdown('0/0')

    with c1:
        st.empty()
        use_webcam = st.sidebar.toggle("Use Webcam")

        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
                width: 350px
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
                width: 350px
                margin-left: -350px
            }
            </style>


            """,
            unsafe_allow_html=True,
        )

        st.sidebar.markdown("---")

        stframe = st.empty()

        if not use_webcam:
            vid = cv2.VideoCapture(DEMO_VIDEO)
        else:
            vid = cv2.VideoCapture(0)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))
        print(fps_input)

        fps = 0
        i = 0  # iterations

        st.markdown("<hr/>", unsafe_allow_html=True)

        ## Dashboard
        detector = FrozenShoulder("20250106v1_45stepsize.h5")

        prevTime = 0
        warmup_frames = 60
        frame_count = 0

        while vid.isOpened():
            i += 1
            success, img = vid.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.pose.process(imgRGB)
            img = detector.findPose(img, False)
            lmList = detector.findPosition(img, False)
            if len(lmList) != 0:
                degree_of_movement = detector.angle(img, lmList[11][1:], lmList[13][1:], lmList[12][1:], lmList[14][1:])
                current_state = detector.get_state(degree_of_movement)
                detector.update_state_sequence(current_state)
                L, R = detector.counter(img, current_state)
            frame_count += 1

            if frame_count > warmup_frames:
                img = detector.process_frame(img, results)

            # FPS Counter Logic
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime

            #  Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color:red;'>{int(degree_of_movement)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color:red;'>{width}</h1>", unsafe_allow_html=True)
            count_left_shoulder.write("**SHOULDER LEFT: {}**".format(L))
            count_right_shoulder.write("**SHOULDER LEFT: {}**".format(R))

        imgRGB = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
        imgRGB = image_resize(image=img, width=640)
        stframe.image(img, channels="BGR", use_container_width=True)

































