import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp
import streamlit as st
import time
import base64
from collections import deque
import streamlit.components.v1 as components

# Calculate eye aspect ratio (EAR)
def calculate_ear(eye):
    if len(eye) != 6:
        return 0
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    if C == 0:
        return 0
    ear = (A + B) / (2.0 * C)
    return ear

# Function to play alert sound using an HTML audio element
def play_alert_sound():
    try:
        # Ensure you have an 'alert.wav' file in your project directory.
        with open("alert.wav", "rb") as audio_file:
            audio_bytes = audio_file.read()
            b64_audio = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
            </audio>
            """
            # This will inject the HTML that auto-plays the audio
            components.html(audio_html, height=0)
    except Exception as e:
        st.error(f"Error playing alert sound: {e}")

# Constants
EAR_THRESHOLD = 0.3  # Adjusted to detect partial eye closure
CONSECUTIVE_FRAMES = 10  # Faster response to partial closure

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Streamlit UI styling and title
st.markdown("""
    <style>
        .main {background-color: #E6F0FF;}
        h1 {color: #0056b3; text-align: center;}
        .stButton>button {background-color: #0056b3; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title('Driver Drowsiness Detection')
run = st.checkbox('Start Detection')

# Initialize counters and logs
frame_count = 0
alert_triggered = False  # Flag to avoid repeated alerts during one drowsy event

drowsiness_log = []           # Log for drowsiness events
ear_values = deque(maxlen=100)  # Store EAR values for graphing
screenshot_gallery = []       # To store screenshots when alert is triggered

cap = cv2.VideoCapture(0)

if run:
    stframe = st.empty()
    graph = st.line_chart([])
    log_placeholder = st.sidebar.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        drowsiness_active = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                try:
                    left_eye = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) 
                                for i in [362, 385, 387, 263, 373, 380]]
                    right_eye = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) 
                                 for i in [33, 160, 158, 133, 153, 144]]

                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    ear_values.append(ear)
                    graph.line_chart(list(ear_values))

                    if ear < EAR_THRESHOLD:
                        frame_count += 1
                        if frame_count >= CONSECUTIVE_FRAMES and not alert_triggered:
                            alert_triggered = True
                            drowsiness_active = True

                            # Draw alert on frame
                            cv2.putText(frame, 'DROWSINESS DETECTED!', (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                            # Log the event and take a screenshot
                            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                            drowsiness_log.append({'Timestamp': timestamp, 'Status': 'Drowsiness Detected'})
                            screenshot = frame.copy()
                            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                            screenshot_gallery.append({'timestamp': timestamp, 'image': screenshot})
                            st.sidebar.image(screenshot, caption=f"Captured at {timestamp}", use_column_width=True)

                            # Play alert sound and show dialog box alert
                            play_alert_sound()
                            st.error("Drowsiness Alert! Please take a break!")
                    else:
                        frame_count = 0
                        alert_triggered = False

                    # Draw eye landmarks
                    for (x, y) in left_eye + right_eye:
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    # Draw face bounding box
                    ih, iw, _ = frame.shape
                    bbox_x = int(min([lm.x for lm in landmarks]) * iw)
                    bbox_y = int(min([lm.y for lm in landmarks]) * ih)
                    bbox_w = int((max([lm.x for lm in landmarks]) - min([lm.x for lm in landmarks])) * iw)
                    bbox_h = int((max([lm.y for lm in landmarks]) - min([lm.y for lm in landmarks])) * ih)
                    cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (255, 0, 0), 2)

                    if drowsiness_active:
                        cv2.putText(frame, 'DROWSINESS ACTIVE!', (bbox_x, bbox_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                except Exception as e:
                    st.error(f"Error processing face landmarks: {e}")

        stframe.image(frame, channels="BGR")

        if drowsiness_log:
            log_placeholder.table(drowsiness_log)

        # Allow stopping the loop if the checkbox is unchecked
        if not run:
            break

cap.release()

# Sidebar navigation for logs and screenshot gallery
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Drowsiness Log", "Screenshot Gallery"])

if page == "Drowsiness Log" and drowsiness_log:
    st.subheader("Drowsiness Log")
    st.table(drowsiness_log)

elif page == "Screenshot Gallery" and screenshot_gallery:
    st.subheader("Drowsiness Screenshots")
    for item in screenshot_gallery:
        st.image(item['image'], caption=item['timestamp'], use_column_width=True)
