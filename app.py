import streamlit as st
import cv2
from deepface import DeepFace
from PIL import Image
import time

# Emoji mapping
emotion_emojis = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("😊 Real-Time Emotion Detector")
st.markdown("Use the buttons to start or stop emotion detection.")

# Initialize session state variable if not present
if 'running' not in st.session_state:
    st.session_state.running = False

# Buttons to control detection
start_btn = st.button("Start Detection")
stop_btn = st.button("Stop Detection")

if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

frame_window = st.image([])
status_text = st.empty()

def detect_emotion():
    cap = cv2.VideoCapture(0)
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not accessible!")
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant = result[0]['dominant_emotion']
            emoji = emotion_emojis.get(dominant, "")
            status_text.markdown(f"### Detected Emotion: **{dominant.capitalize()}** {emoji}")
        except:
            status_text.markdown("### No face detected 😶")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        frame_window.image(img)

        time.sleep(0.5)

    cap.release()
    status_text.markdown("### Detection stopped.")

# Run detection if running is True
if st.session_state.running:
    detect_emotion()
