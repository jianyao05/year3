import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# Read from webcam
video = "C:\Users\223162D\PycharmProjects\year3\movement_datasets\SHOULDER_FLEXION_RIGHT\SHOULDER_FLEXION_RIGHT_0-50.avi_output.avi"
cap = cv2.VideoCapture(video)

# Initialize mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Parameters
#label = "ARMPIT_LEFT"  # Label for the dataset (change for specific movement)
#label = "CIRCLE_LEFT"  # Label for the dataset (change for specific movement)
#label = "CIRCLE_RIGHT"  # Label for the dataset (change for specific movement)
#label = "CROSS_BODY_LEFT"  # Label for the dataset (change for specific movement)
#label = "CROSS_BODY_RIGHT"  # Label for the dataset (change for specific movement)
#label = "PENDULUM_LEFT"  # Label for the dataset (change for specific movement)
#label = "PENDULUM_RIGHT"  # Label for the dataset (change for specific movement)
#label = "SHOULDER_FLEXION_LEFT"  # Label for the dataset (change for specific movement) get yuhang to do more examples
label = "SHOULDER_FLEXION_RIGHT"  # Label for the dataset (change for specific movement)

n_repetitions_to_add = 50  # Number of repetitions to add
n_time_steps_per_rep = int(fps * 2)  # Number of frames per repetition (increased for longer capture)
save_folder = f"./recorded_movement_datasets/{label}/"  # Folder to save repetitions
lm_list = []  # To store landmarks for one repetition
frame_count = 0  # Track the frame count per repetition
break_time = 0.5  # Break time between repetitions in seconds

# Create folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Determine the starting repetition number by checking existing files
existing_files = [f for f in os.listdir(save_folder) if f.endswith(".csv")]
existing_reps = [int(f.split('_rep')[1].split('.')[0]) for f in existing_files if '_rep' in f]
next_rep_start = max(existing_reps, default=0) + 1  # Start from the next available number

# Initialize VideoWriter
output_filename = f"{label}_{len(existing_reps)}-{n_repetitions_to_add+len(existing_reps)}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(f'{save_folder}/{output_filename}_output.avi', fourcc, fps, (frame_width, frame_height))

def normalize_landmarks(landmarks):
    # Get the reference point (e.g., the nose or a fixed central landmark like the hip or shoulder)
    ref_lm = landmarks[mpPose.PoseLandmark.NOSE]

    ref_x = ref_lm.x
    ref_y = ref_lm.y
    ref_z = ref_lm.z

    normalized_lm = []

    # Loop through all landmarks and normalize relative to the reference landmark (e.g., nose)
    for lm in landmarks:
        normalized_x = lm.x - ref_x
        normalized_y = lm.y - ref_y
        normalized_z = lm.z - ref_z
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


def draw_landmark_on_image(mpDraw, results, img):
    """Draw landmarks and connections on image."""
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    for lm in results.pose_landmarks.landmark:
        h, w, _ = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    return img




# Main loop for capturing and processing data
for rep_count in range(next_rep_start, next_rep_start + n_repetitions_to_add):
    print(f"Starting repetition {rep_count} of {next_rep_start + n_repetitions_to_add - 1}...")
    time.sleep(1)  # Brief pause before starting

    while frame_count < n_time_steps_per_rep:
        ret, frame = cap.read()
        if ret:
            # Process frame for pose detection
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)

            if results.pose_landmarks:
                lm = make_landmark_timestep(results)

                if lm:  # If leg landmarks were successfully extracted
                    lm_list.append(lm)
                    frame_count += 1

                    # Draw pose landmarks on image
                    frame = draw_landmark_on_image(mpDraw, results, frame)

            text = f"Recording Frame: {rep_count}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (255, 255, 255)  # White text
            thickness = 2

            # Get text size for the rectangle
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = 10
            text_y = 30
            box_coords = ((text_x - 5, text_y - 20), (text_x + text_size[0] + 5, text_y + 5))

            # Draw the rectangle and the text
            cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)  # Black rectangle
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

            # Write frame to the output video
            output_video.write(frame)

            # Display the frame with pose landmarks
            cv2.imshow("image", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    # Save the repetition as a CSV file
    df = pd.DataFrame(lm_list)
    df.to_csv(f"{save_folder}/{label}_rep{rep_count}.csv", index=False)
    print(f"Saved repetition {rep_count} to {save_folder}")

    # Reset for the next repetition
    lm_list = []
    frame_count = 0

    # Take a break before the next repetition
    if rep_count < next_rep_start + n_repetitions_to_add - 1:
        print(f"Take a {break_time} second break before starting the next repetition.")
        time.sleep(break_time)

# Final message
print(f"Added {n_repetitions_to_add} repetitions to the dataset.")

# Release the webcam and VideoWriter, and close OpenCV windows
cap.release()
output_video.release()
cv2.destroyAllWindows()
