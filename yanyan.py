
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

import streamlit as st


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

        self.label = "Warmup...."       # types of label for each exercise
        self.lm_list = []  # Store landmark data for prediction

        # Exercise state tracking
        self.list = []
        self.state_sequence = []

        self.counter_left_flexion = 0
        self.counter_right_flexion = 0
        self.counter_left_cross = 0
        self.counter_right_cross = 0

       # TARGETS
        ###
        self.target_left_armpit = 0
        self.target_right_armpit = 0
        ###
        self.target_left_circle = 0
        self.target_right_circle = 0
        ###
        self.target_left_cross = 0
        self.target_right_cross = 0
        ###
        self.target_left_pendulum = 0
        self.target_right_pendulum = 0
        ###
        self.target_left_flexion = 0
        self.target_right_flexion = 0
        ###
        self.target_left_towel = 0
        self.target_right_towel = 0
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
        cv2.putText(img, "L-Flexion: {}".format(str(self.counter_left_flexion)), (510, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "R-Flexion: {}".format(str(self.counter_right_flexion)), (510, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        return img

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
        elif self.label == "CIRCLE LEFT":
            cv2.line(img, (LS[0], 400), LS[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.line(img, LS[:2], LE[:2], (102, 204, 255), 4, cv2.LINE_AA)
            cv2.circle(img, LS[:2], 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, LE[:2], 5, (255, 0, 0), cv2.FILLED)
            reference = LS[0], 1000, LS[2]
            vector1 = np.array(reference) - np.array(LS)
            vector2 = np.array(LE) - np.array(LS)
        elif self.label == "CIRCLE RIGHT":
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

        if 0 <= angle <= 30:
            state = 1
        elif 35 <= angle <= 75:
            state = 2
        elif 80 <= angle <= 100:
            state = 3
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
                    self.counter_left_flexion += 1
                elif self.label == "FLEXION RIGHT":
                    self.counter_right_flexion += 1
                elif self.label == "CIRCLE LEFT":
                    self.counter_left_cross += 1
                elif self.label == "CIRCLE RIGHT":
                    self.counter_right_cross += 1
                else:
                    pass

            elif 's2' in self.list and len(self.list) == 1:
                print("Improper Form")

            self.list = []

        return self.counter_left_flexion, self.counter_right_flexion


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


### ------------------------------------------------ STATE SESSIONS ------------------------------------------------ ###
# Armpit Stretch
if "target_left_armpit" not in st.session_state:
    st.session_state.target_left_armpit = 0
if "target_right_armpit" not in st.session_state:
    st.session_state.target_right_armpit = 0

# Arm Circles
if "target_left_circle" not in st.session_state:
    st.session_state.target_left_circle = 0
if "target_right_circle" not in st.session_state:
    st.session_state.target_right_circle = 0

# Cross Body Stretch
if "target_left_cross" not in st.session_state:
    st.session_state.target_left_cross = 0
if "target_right_cross" not in st.session_state:
    st.session_state.target_right_cross = 0

# Pendulum Swing
if "target_left_pendulum" not in st.session_state:
    st.session_state.target_left_pendulum = 0
if "target_right_pendulum" not in st.session_state:
    st.session_state.target_right_pendulum = 0

# Shoulder Flexion
if "target_left_flexion" not in st.session_state:
    st.session_state.target_left_flexion = 0
if "target_right_flexion" not in st.session_state:
    st.session_state.target_right_flexion = 10

# Towel Stretch
if "target_left_towel" not in st.session_state:
    st.session_state.target_left_towel = 0
if "target_right_towel" not in st.session_state:
    st.session_state.target_right_towel = 0

### ------------------------------------ START OF USER INTERFACE CUSTOMISATIONS ------------------------------------ ###
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

app_mode = st.sidebar.selectbox("Choose the App Mode", ["Target", "Video"])
# Define a function to handle the "Video" page
def render_video_page():
    st.title("Frozen Shoulder Rehabilitation - Video")
    use_webcam = st.sidebar.toggle("Use Webcam")

    vid = cv2.VideoCapture(0)
    detector = FrozenShoulder("20250106v1_45stepsize.h5")

    # Assign targets from session state
    detector.target_left_flexion = st.session_state.target_left_flexion
    detector.target_right_flexion = st.session_state.target_right_flexion

    i = 0  # iterations
    warmup_frames = 60
    frame_count = 0

    # Create the layout
    c1, c2 = st.columns([0.7, 0.3], border=True)

    with c2:
        st.write(
            """
            <div style='text-align: center;'>
                <h4 style='text-decoration: underline; font-weight: bold;'>TARGET</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Target
        text1 = st.markdown("Left Shoulder Flexion: 3 / 10 Sets")
        text2 = st.markdown("Right Shoulder Flexion: 10 / 10 Sets [Completed]")

    # Video display goes in c1
    with c1:
        exercise_label = st.markdown("Type of Exercise")
        stframe = st.empty()  # Video frame placeholder

    # Process video frames
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
            detector.counter(img, current_state)

            # Update the counters in the UI
            with c2:
                if detector.label == "FLEXION LEFT":
                    color_left_flexion = "red"
                    color_right_flexion = "black"
                elif detector.label == "FLEXION RIGHT":
                    color_left_flexion = "black"
                    color_right_flexion = "red"
                else:
                    color_left_flexion = "black"
                    color_right_flexion = "black"

                text1.write(
                    f"""
                    <div style='display: flex; justify-content: space-between; align-items: left;'>
                        <h5 style='color: {color_left_flexion};'>Left Shoulder Flexion: {detector.counter_left_flexion} / {st.session_state.target_left_flexion}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                text2.write(
                    f"""
                    <div style='display: flex; justify-content: space-between; align-items: left;'>
                        <h5 style='color: {color_right_flexion};'>Right Shoulder Flexion: {detector.counter_right_flexion} / {st.session_state.target_right_flexion}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Display the processed frame in c1
        imgRGB = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)
        imgRGB = image_resize(image=img, width=640)
        with c1:
            exercise_label.write(
                f"""
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h3 style='color: red;'>Current Exercise: {detector.label}</h3>
                    <h3 style='color: blue;'>ROM: {int(degree_of_movement)}°</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
            stframe.image(imgRGB, channels="BGR", use_container_width=True)

        frame_count += 1
        if frame_count > warmup_frames:
            img = detector.process_frame(img, results)


# Define a function to handle the "Target" page
def render_target_page():
    st.title("Set Exercise Targets")
    T1, T2, T3 = st.columns(3, border=True)
    with T1:
        st.subheader("Armpit Stretch")
        st.session_state.target_left_armpit = st.number_input(
            "Target Repetitions for Left Armpit Stretch",
            step=1,
            value=st.session_state.target_left_armpit,
            placeholder="Enter Amount..."
        )
        st.session_state.target_right_armpit = st.number_input(
            "Target Repetitions for Right Armpit Stretch",
            step=1,
            value=st.session_state.target_right_armpit,
            placeholder="Enter Amount..."
        )
    with T2:
        st.subheader("Arm Circles")
        st.session_state.target_left_circle = st.number_input(
            "Target Repetitions for Left Arm Circles",
            step=1,
            value=st.session_state.target_left_circle,
            placeholder="Enter Amount..."
        )
        st.session_state.target_right_circle = st.number_input(
            "Target Repetitions for Right Arm Circles",
            step=1,
            value=st.session_state.target_right_circle,
            placeholder="Enter Amount..."
        )
    with T3:
        st.subheader("Cross Body Stretch")
        st.session_state.target_left_cross = st.number_input(
            "Target Repetitions for Left Cross Body Stretch",
            step=1,
            value=st.session_state.target_left_cross,
            placeholder="Enter Amount..."
        )
        st.session_state.target_right_cross = st.number_input(
            "Target Repetitions for Right Cross Body Stretch",
            step=1,
            value=st.session_state.target_right_cross,
            placeholder="Enter Amount..."
        )


# Main rendering logic
if __name__ == "__main__":
    app_mode = st.sidebar.selectbox("Choose the App Mode", ["Target", "Video"])
    if app_mode == "Target":
        render_target_page()
    elif app_mode == "Video":
        render_video_page()

