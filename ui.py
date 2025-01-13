import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time

DEMO_VIDEO = "movement_datasets\\ARMPIT_RIGHT\\ARMPIT_RIGHT_0-50.avi_output.avi"

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
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width / float(w)
        dim = int(w*r), height
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # reizing of image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

app_mode = st.sidebar.selectbox("Choose the App Mode",
                                ["About Application", "Real-Time Video Interface", "About Us"])

if app_mode == "About Application":
    st.markdown("In this Application we are targetting Frozen Shoulder Patients. Using **MediaPipe** to extract key landmarks from the body and arms.")


elif app_mode == "Real-Time Video Interface":
    
    use_webcam = st.sidebar.button("Use Webcam")
    record = st.sidebar.checkbox("Record Video")

    if record:
        st.checkbox("Recording", value = True)

    st.sidebar.markdown("---")
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


    max_faces = st.sidebar.number_input("Maximum Number of Faces?? CHANGE THIS ASAP", value = 2, min_value = 1)
    st.sidebar.markdown("---")
    detection_confidence = st.sidebar.slider("Min Detection Confidence", min_value = 0.0, max_value = 1.0, value = 0.5)
    tracking_confidence = st.sidebar.slider("Min Tracking Confidence", min_value = 0.0, max_value = 1.0, value = 0.5)

    st.sidebar.markdown("---")

    st.markdown("## Output")

    stframe = st.empty
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown('0')

    with kpi2:
        st.markdown('**Amount of Proper Squat**')
        kpi2_text = st.markdown('0')

    with kpi3:
        st.markdown('**Range of Motion**')
        kpi3_text = st.markdown('0')

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    # Recording Part
    codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text("Input Video")
    st.sidebar.video(tffile.name)

    



elif app_mode == "Run on Video":
    st.sidebar.markdown("---")
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


    max_faces = st.sidebar.number_input("Maximum Number of Faces?? CHANGE THIS ASAP", value = 2, min_value = 1)
    st.sidebar.markdown("---")
    detection_confidence = st.sidebar.slider("Min Detection Confidence", min_value = 0.0, max_value = 1.0, value = 0.5)
    tracking_confidence = st.sidebar.slider("Min Tracking Confidence", min_value = 0.0, max_value = 1.0, value = 0.5)

    st.sidebar.markdown("---")

